// Per-pixel letterbox arithmetic, shared by the CUDA kernel (preprocess.cu) and the CPU build used by
// the desktop test, so both compute exactly the same numbers.
//
// Output = the model input: RGB, planar (CHW), float in [0, 1], the camera picture resized
// (bilinear, OpenCV INTER_LINEAR convention, rounded to 8 bits like cv::resize on uint8) and centred,
// the rest grey 114 -- what Ultralytics' LetterBox(auto=False) + /255 produce.
// For the 1280x720 stream into 640-wide inputs the scale is exactly 1/2, where this equals
// OpenCV bit for bit (2x2 average with rounding, OpenCV's INTER_AREA fast path).
//
// Input frames: BGR (3 bytes per pixel) or NV12 (Y plane + interleaved UV plane at half
// resolution, straight from the hardware decoder). NV12 pixels are converted with OpenCV's
// COLOR_YUV2BGR_NV12 integer formula (BT.601, video range) before the bilinear mix, so NV12 input
// gives the same result as cv::cvtColor(NV12 -> BGR) followed by the BGR path.
#pragma once

#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define PT_HD __host__ __device__ __forceinline__
#else
#define PT_HD inline
#endif

namespace pt {

enum class PixelFormat : int { BGR = 0, NV12 = 1 };

constexpr int kPadValue = 114;

// Ultralytics LetterBox(auto=False, center=True) geometry, and scale_boxes() for the way back.
struct Letterbox {
    int src_w, src_h;        // camera frame
    int dst_w, dst_h;        // model input
    int new_w, new_h;        // resized picture inside the input
    int left, top;           // padding before the picture
    float scale_x, scale_y;  // src / new  (sampling step, as cv::resize computes it)
    float gain_x, gain_y;    // new / src  (to map boxes back)
};

// Host only (std::nearbyint = Python's round(): halves go to even).
inline Letterbox make_letterbox(int src_w, int src_h, int dst_w, int dst_h) {
    Letterbox lb{};
    lb.src_w = src_w;
    lb.src_h = src_h;
    lb.dst_w = dst_w;
    lb.dst_h = dst_h;
    const double r = std::fmin(static_cast<double>(dst_h) / src_h, static_cast<double>(dst_w) / src_w);
    lb.new_w = static_cast<int>(std::nearbyint(src_w * r));
    lb.new_h = static_cast<int>(std::nearbyint(src_h * r));
    const double dw = (dst_w - lb.new_w) / 2.0, dh = (dst_h - lb.new_h) / 2.0;
    lb.left = static_cast<int>(std::nearbyint(dw - 0.1));
    lb.top = static_cast<int>(std::nearbyint(dh - 0.1));
    lb.scale_x = static_cast<float>(static_cast<double>(src_w) / lb.new_w);
    lb.scale_y = static_cast<float>(static_cast<double>(src_h) / lb.new_h);
    lb.gain_x = static_cast<float>(static_cast<double>(lb.new_w) / src_w);
    lb.gain_y = static_cast<float>(static_cast<double>(lb.new_h) / src_h);
    return lb;
}

// Box mapping back to camera pixels when the frame given to the letterbox was already scaled down from
// the camera's cam_w x cam_h (hardware resize): same picture placement inside the model input, gains
// relative to the camera size. With cam == frame size this returns lb unchanged.
inline Letterbox for_camera(Letterbox lb, int cam_w, int cam_h) {
    lb.gain_x = static_cast<float>(static_cast<double>(lb.new_w) / cam_w);
    lb.gain_y = static_cast<float>(static_cast<double>(lb.new_h) / cam_h);
    lb.src_w = cam_w;
    lb.src_h = cam_h;
    lb.scale_x = static_cast<float>(static_cast<double>(cam_w) / lb.new_w);
    lb.scale_y = static_cast<float>(static_cast<double>(cam_h) / lb.new_h);
    return lb;
}

// Source sample positions for one output coordinate (cv::resize INTER_LINEAR, edge clamped).
struct Tap {
    int i0, i1;
    float f;  // weight of i1
};

PT_HD Tap make_tap(int d, float scale, int size) {
    float f = (static_cast<float>(d) + 0.5f) * scale - 0.5f;
    int i = static_cast<int>(f >= 0.f ? f : f - 1.f);  // floor (f > -1 here)
    f -= static_cast<float>(i);
    if (i < 0) {
        i = 0;
        f = 0.f;
    }
    if (i >= size - 1) {
        i = size - 1;
        f = 0.f;
    }
    Tap t;
    t.i0 = i;
    t.i1 = i + 1 < size ? i + 1 : i;
    t.f = f;
    return t;
}

PT_HD float round8(float v) {  // round to the nearest 8-bit level, like cv::resize writing uint8
    float r = static_cast<float>(static_cast<int>(v + 0.5f));
    return r > 255.f ? 255.f : r;
}

PT_HD int clamp255(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }

// OpenCV COLOR_YUV2BGR_NV12 (color_yuv.simd.hpp: ITUR_BT_601_* fixed point, 20 bits).
PT_HD void nv12_pixel_bgr(const uint8_t* y_plane, int y_stride, const uint8_t* uv_plane, int uv_stride, int px,
                          int py, int bgr[3]) {
    const int yv = y_plane[py * y_stride + px];
    const uint8_t* uv = uv_plane + (py >> 1) * uv_stride + (px & ~1);
    const int u = static_cast<int>(uv[0]) - 128, v = static_cast<int>(uv[1]) - 128;
    const int ruv = (1 << 19) + 1673527 * v;
    const int guv = (1 << 19) - 852492 * v - 409993 * u;
    const int buv = (1 << 19) + 2116026 * u;
    const int y = (yv - 16 > 0 ? yv - 16 : 0) * 1220542;
    bgr[0] = clamp255((y + buv) >> 20);
    bgr[1] = clamp255((y + guv) >> 20);
    bgr[2] = clamp255((y + ruv) >> 20);
}

// Writes output pixel (x, y) of the planar RGB float input.
// BGR: src = frame, stride = bytes per row.  NV12: src = Y plane (UV plane follows after src_h rows),
// stride = bytes per row of both planes.
PT_HD void letterbox_pixel(const uint8_t* src, int stride, PixelFormat fmt, const Letterbox& lb, float* dst, int x,
                           int y) {
    const int plane = lb.dst_w * lb.dst_h;
    float* o = dst + y * lb.dst_w + x;
    const int rx = x - lb.left, ry = y - lb.top;
    if (rx < 0 || ry < 0 || rx >= lb.new_w || ry >= lb.new_h) {
        const float pad = static_cast<float>(kPadValue) / 255.f;
        o[0] = o[plane] = o[2 * plane] = pad;
        return;
    }
    const Tap tx = make_tap(rx, lb.scale_x, lb.src_w);
    const Tap ty = make_tap(ry, lb.scale_y, lb.src_h);
    int p00[3], p01[3], p10[3], p11[3];  // B, G, R
    if (fmt == PixelFormat::BGR) {
        const uint8_t* r0 = src + ty.i0 * stride;
        const uint8_t* r1 = src + ty.i1 * stride;
        for (int c = 0; c < 3; ++c) {
            p00[c] = r0[tx.i0 * 3 + c];
            p01[c] = r0[tx.i1 * 3 + c];
            p10[c] = r1[tx.i0 * 3 + c];
            p11[c] = r1[tx.i1 * 3 + c];
        }
    } else {
        const uint8_t* uv = src + lb.src_h * stride;
        nv12_pixel_bgr(src, stride, uv, stride, tx.i0, ty.i0, p00);
        nv12_pixel_bgr(src, stride, uv, stride, tx.i1, ty.i0, p01);
        nv12_pixel_bgr(src, stride, uv, stride, tx.i0, ty.i1, p10);
        nv12_pixel_bgr(src, stride, uv, stride, tx.i1, ty.i1, p11);
    }
    const float fx = tx.f, fy = ty.f;
    const float w00 = (1.f - fx) * (1.f - fy), w01 = fx * (1.f - fy), w10 = (1.f - fx) * fy, w11 = fx * fy;
    for (int c = 0; c < 3; ++c) {
        const float v = w00 * static_cast<float>(p00[c]) + w01 * static_cast<float>(p01[c]) +
                        w10 * static_cast<float>(p10[c]) + w11 * static_cast<float>(p11[c]);
        o[(2 - c) * plane] = round8(v) / 255.f;  // source B,G,R -> planes R,G,B
    }
}

}  // namespace pt
