// Per-pixel input arithmetic, shared by the CUDA kernels (preprocess.cu) and the CPU build used by the
// desktop tests, so both compute exactly the same numbers.
//
// Two ways a model input is cut from the camera frame:
//   letterbox  a rectangle of the frame (the whole frame, or a region) resized to fit the input and centred,
//              the rest padded; bilinear with OpenCV's INTER_LINEAR convention, rounded to 8 bits like
//              cv::resize on uint8 (exactly cv::resize INTER_LINEAR_EXACT at 1/2 and 1/4 scale).
//              Used for the face detector (whole frame) and the palm detector (region around the target).
//   rotcrop    a rotated square of the frame (a hand), sampled straight into the input: bilinear, averaged
//              over n x n points per output pixel when shrinking (no aliasing), black outside the frame.
// Output layout and values per model (InputFormat): NCHW or NHWC, RGB or BGR, scale 1 (0..255) or 1/255.
//
// Frame coordinates are continuous: pixel (i, j) covers [i, i+1) x [j, j+1), so its centre is (i+0.5, j+0.5).
// Frames: BGR (3 bytes per pixel) or NV12 (Y plane, then the interleaved UV plane at half resolution);
// NV12 pixels are converted with OpenCV's COLOR_YUV2BGR_NV12 integer formula first.
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

// How a model wants its input.
struct InputFormat {
    bool nhwc = false;    // false: planar (1x3xHxW); true: interleaved (1xHxWx3)
    bool rgb = true;      // channel order
    float scale = 1.f / 255.f;
    float pad = 0.f;      // value of padding / outside-the-frame pixels (already scaled)
};

// A region of the frame resized to fit the input, centred (Ultralytics LetterBox geometry).
struct Letterbox {
    int src_x0, src_y0;      // region origin in the frame
    int src_w, src_h;        // region size
    int dst_w, dst_h;        // model input
    int new_w, new_h;        // resized region inside the input
    int left, top;           // padding before it
    float scale_x, scale_y;  // src / new (sampling step)
    float gain_x, gain_y;    // new / src (input pixels per frame pixel)
};

// Host only (std::nearbyint = Python's round(): halves go to even).
inline Letterbox make_letterbox(int src_x0, int src_y0, int src_w, int src_h, int dst_w, int dst_h) {
    Letterbox lb{};
    lb.src_x0 = src_x0;
    lb.src_y0 = src_y0;
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
inline Letterbox make_letterbox(int src_w, int src_h, int dst_w, int dst_h) {
    return make_letterbox(0, 0, src_w, src_h, dst_w, dst_h);
}

// Input pixel coordinates -> frame coordinates (and back).
PT_HD float lb_to_frame_x(const Letterbox& lb, float x) { return (x - lb.left) / lb.gain_x + lb.src_x0; }
PT_HD float lb_to_frame_y(const Letterbox& lb, float y) { return (y - lb.top) / lb.gain_y + lb.src_y0; }

// A rotated square of the frame: centre, side length (frame pixels) and the angle (radians) by which the
// square's "up" is turned clockwise from the image's up. Square point (dx, dy), measured from the centre
// with dy pointing to the square's bottom, is frame point
//   (cx + dx cos a - dy sin a,  cy + dx sin a + dy cos a).
struct RotRect {
    float cx, cy, size, angle;
};

PT_HD void rot_to_frame(const RotRect& r, float dx, float dy, float& x, float& y) {
    const float c = cosf(r.angle), s = sinf(r.angle);
    x = r.cx + dx * c - dy * s;
    y = r.cy + dx * s + dy * c;
}
PT_HD void frame_to_rot(const RotRect& r, float x, float y, float& dx, float& dy) {
    const float c = cosf(r.angle), s = sinf(r.angle);
    const float ux = x - r.cx, uy = y - r.cy;
    dx = ux * c + uy * s;
    dy = -ux * s + uy * c;
}
// Model input pixel (u, v) of an out x out crop (continuous coordinates, 0..out) -> frame.
PT_HD void crop_to_frame(const RotRect& r, int out, float u, float v, float& x, float& y) {
    rot_to_frame(r, (u / out - 0.5f) * r.size, (v / out - 0.5f) * r.size, x, y);
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

// One frame pixel as B, G, R. frame_h: rows of the frame (NV12: the UV plane follows after them).
PT_HD void frame_pixel(const uint8_t* src, int stride, PixelFormat fmt, int frame_h, int px, int py, int bgr[3]) {
    if (fmt == PixelFormat::BGR) {
        const uint8_t* p = src + py * stride + px * 3;
        bgr[0] = p[0];
        bgr[1] = p[1];
        bgr[2] = p[2];
    } else {
        nv12_pixel_bgr(src, stride, src + frame_h * stride, stride, px, py, bgr);
    }
}

// Writes the 3 values of input pixel (x, y) (given as B, G, R floats) in the model's layout and order.
PT_HD void put_pixel(float* dst, int dst_w, int dst_h, const InputFormat& in, int x, int y, float b, float g, float r) {
    const float c0 = in.rgb ? r : b, c2 = in.rgb ? b : r;
    if (in.nhwc) {
        float* o = dst + (y * dst_w + x) * 3;
        o[0] = c0 * in.scale;
        o[1] = g * in.scale;
        o[2] = c2 * in.scale;
    } else {
        const int plane = dst_w * dst_h;
        float* o = dst + y * dst_w + x;
        o[0] = c0 * in.scale;
        o[plane] = g * in.scale;
        o[2 * plane] = c2 * in.scale;
    }
}

// Letterbox: writes input pixel (x, y). frame_h: rows of the whole frame (for NV12).
PT_HD void letterbox_pixel(const uint8_t* src, int stride, PixelFormat fmt, int frame_h, const Letterbox& lb,
                           const InputFormat& in, float* dst, int x, int y) {
    const int rx = x - lb.left, ry = y - lb.top;
    if (rx < 0 || ry < 0 || rx >= lb.new_w || ry >= lb.new_h) {
        const float p = in.scale != 0.f ? in.pad / in.scale : 0.f;  // put_pixel scales again
        put_pixel(dst, lb.dst_w, lb.dst_h, in, x, y, p, p, p);
        return;
    }
    const Tap tx = make_tap(rx, lb.scale_x, lb.src_w);
    const Tap ty = make_tap(ry, lb.scale_y, lb.src_h);
    int p00[3], p01[3], p10[3], p11[3];
    frame_pixel(src, stride, fmt, frame_h, lb.src_x0 + tx.i0, lb.src_y0 + ty.i0, p00);
    frame_pixel(src, stride, fmt, frame_h, lb.src_x0 + tx.i1, lb.src_y0 + ty.i0, p01);
    frame_pixel(src, stride, fmt, frame_h, lb.src_x0 + tx.i0, lb.src_y0 + ty.i1, p10);
    frame_pixel(src, stride, fmt, frame_h, lb.src_x0 + tx.i1, lb.src_y0 + ty.i1, p11);
    const float fx = tx.f, fy = ty.f;
    const float w00 = (1.f - fx) * (1.f - fy), w01 = fx * (1.f - fy), w10 = (1.f - fx) * fy, w11 = fx * fy;
    float v[3];
    for (int c = 0; c < 3; ++c)
        v[c] = round8(w00 * static_cast<float>(p00[c]) + w01 * static_cast<float>(p01[c]) +
                      w10 * static_cast<float>(p10[c]) + w11 * static_cast<float>(p11[c]));
    put_pixel(dst, lb.dst_w, lb.dst_h, in, x, y, v[0], v[1], v[2]);
}

// Bilinear sample at continuous frame point (x, y); false if outside the frame.
PT_HD bool sample_bilinear(const uint8_t* src, int stride, PixelFormat fmt, int frame_w, int frame_h, float x, float y,
                           float out[3]) {
    const float fx = x - 0.5f, fy = y - 0.5f;
    if (fx < -0.5f || fy < -0.5f || fx > frame_w - 0.5f || fy > frame_h - 0.5f) return false;
    int x0 = static_cast<int>(floorf(fx)), y0 = static_cast<int>(floorf(fy));
    float ax = fx - x0, ay = fy - y0;
    int x1 = x0 + 1, y1 = y0 + 1;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > frame_w - 1) x1 = frame_w - 1;
    if (y1 > frame_h - 1) y1 = frame_h - 1;
    int p00[3], p01[3], p10[3], p11[3];
    frame_pixel(src, stride, fmt, frame_h, x0, y0, p00);
    frame_pixel(src, stride, fmt, frame_h, x1, y0, p01);
    frame_pixel(src, stride, fmt, frame_h, x0, y1, p10);
    frame_pixel(src, stride, fmt, frame_h, x1, y1, p11);
    for (int c = 0; c < 3; ++c)
        out[c] = (1.f - ay) * ((1.f - ax) * p00[c] + ax * p01[c]) + ay * ((1.f - ax) * p10[c] + ax * p11[c]);
    return true;
}

// Rotated crop: writes input pixel (u, v) of an out x out input.
PT_HD void rotcrop_pixel(const uint8_t* src, int stride, PixelFormat fmt, int frame_w, int frame_h, const RotRect& r,
                         int out, const InputFormat& in, float* dst, int u, int v) {
    const float step = r.size / out;                  // frame pixels per input pixel
    int n = static_cast<int>(ceilf(step - 0.01f));   // samples per axis when shrinking
    n = n < 1 ? 1 : (n > 4 ? 4 : n);
    float acc[3] = {0.f, 0.f, 0.f};
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            float x, y, s[3];
            crop_to_frame(r, out, u + (i + 0.5f) / n, v + (j + 0.5f) / n, x, y);
            if (sample_bilinear(src, stride, fmt, frame_w, frame_h, x, y, s))
                for (int c = 0; c < 3; ++c) acc[c] += s[c];
        }
    const float k = 1.f / (n * n);  // samples outside the frame count as black (the hand models' padding)
    put_pixel(dst, out, out, in, u, v, acc[0] * k, acc[1] * k, acc[2] * k);
}

}  // namespace pt
