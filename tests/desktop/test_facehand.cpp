// Desktop check of the robot's face + hand code on the CPU (the CUDA kernels run the same per-pixel functions).
// Driven by tests/desktop/check_facehand.py, which compares every result with scripts/facehand_ref.py.
//   letterbox <frame.raw> <w> <h> <bgr|nv12> <x0> <y0> <rw> <rh> <dw> <dh> <nhwc> <rgb> <scale> <out.f32>
//   rotcrop   <frame.raw> <w> <h> <bgr|nv12> <cx> <cy> <size> <angle> <out> <tensor.f32>
//   faces     <out.f32> <x0> <y0> <rw> <rh> <dw> <dh> <fw> <fh> <conf> <result.txt>
//   palms     <out.f32> <x0> <y0> <rw> <rh> <conf> <result.txt>       (each palm + its hand crop)
//   hand      <out.f32> <cx> <cy> <size> <angle> <result.txt>         (landmarks, next crop, finger states, gesture)
//   gesture   <points.txt>                                            (21 lines "x y wx wy wz" -> shape + gesture)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../src/face_detector.hpp"
#include "../../src/gestures.hpp"
#include "../../src/hands.hpp"
#include "../../src/preprocess_math.hpp"

static std::vector<unsigned char> read_file(const char* path) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path);
        std::exit(2);
    }
    std::vector<unsigned char> b;
    unsigned char buf[1 << 16];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) b.insert(b.end(), buf, buf + n);
    std::fclose(f);
    return b;
}

static void write_floats(const char* path, const std::vector<float>& v) {
    std::FILE* f = std::fopen(path, "wb");
    std::fwrite(v.data(), sizeof(float), v.size(), f);
    std::fclose(f);
}

static void print_shape(std::FILE* f, const pt::HandLandmarks& h) {
    const pt::HandShape s = pt::hand_shape(h);
    std::fprintf(f, "shape %d %d %d %d %d %.6f\n", static_cast<int>(s.f[0]), static_cast<int>(s.f[1]),
                 static_cast<int>(s.f[2]), static_cast<int>(s.f[3]), static_cast<int>(s.f[4]), s.touch);
    std::fprintf(f, "gesture %s\n", pt::classify_gesture(h).c_str());
}

int main(int argc, char** argv) {
    const std::string cmd = argc > 1 ? argv[1] : "";
    auto F = [&](int i) { return static_cast<float>(std::atof(argv[i])); };
    auto I = [&](int i) { return std::atoi(argv[i]); };
    if (cmd == "letterbox" && argc >= 16) {
        const auto src = read_file(argv[2]);
        const int w = I(3), h = I(4);
        const pt::PixelFormat fmt = std::string(argv[5]) == "nv12" ? pt::PixelFormat::NV12 : pt::PixelFormat::BGR;
        const pt::Letterbox lb = pt::make_letterbox(I(6), I(7), I(8), I(9), I(10), I(11));
        pt::InputFormat in;
        in.nhwc = I(12) != 0;
        in.rgb = I(13) != 0;
        in.scale = F(14);
        std::vector<float> dst(static_cast<size_t>(3) * lb.dst_w * lb.dst_h);
        const int stride = fmt == pt::PixelFormat::BGR ? w * 3 : w;
        for (int y = 0; y < lb.dst_h; ++y)
            for (int x = 0; x < lb.dst_w; ++x) pt::letterbox_pixel(src.data(), stride, fmt, h, lb, in, dst.data(), x, y);
        write_floats(argv[15], dst);
        return 0;
    }
    if (cmd == "rotcrop" && argc >= 12) {
        const auto src = read_file(argv[2]);
        const int w = I(3), h = I(4), out = I(10);
        const pt::PixelFormat fmt = std::string(argv[5]) == "nv12" ? pt::PixelFormat::NV12 : pt::PixelFormat::BGR;
        const pt::RotRect r{F(6), F(7), F(8), F(9)};
        pt::InputFormat in;
        in.nhwc = true;
        in.rgb = true;
        in.scale = 1.f / 255.f;
        std::vector<float> dst(static_cast<size_t>(3) * out * out);
        const int stride = fmt == pt::PixelFormat::BGR ? w * 3 : w;
        for (int v = 0; v < out; ++v)
            for (int u = 0; u < out; ++u) pt::rotcrop_pixel(src.data(), stride, fmt, w, h, r, out, in, dst.data(), u, v);
        write_floats(argv[11], dst);
        return 0;
    }
    if (cmd == "faces" && argc >= 13) {
        const auto raw = read_file(argv[2]);
        const pt::Letterbox lb = pt::make_letterbox(I(3), I(4), I(5), I(6), I(7), I(8));
        pt::FaceDecoder dec(lb, I(9), I(10), F(11));
        std::vector<pt::Face> faces;
        dec.decode(reinterpret_cast<const float*>(raw.data()), faces);
        std::FILE* f = std::fopen(argv[12], "w");
        for (const auto& fc : faces) {
            std::fprintf(f, "%.9g %.9g %.9g %.9g %.9g", fc.x1, fc.y1, fc.x2, fc.y2, fc.score);
            for (float p : fc.pts) std::fprintf(f, " %.9g", p);
            std::fprintf(f, "\n");
        }
        std::fclose(f);
        return 0;
    }
    if (cmd == "palms" && argc >= 9) {
        const auto raw = read_file(argv[2]);
        const pt::Letterbox lb = pt::make_letterbox(I(3), I(4), I(5), I(6), 192, 192);
        pt::PalmDecoder dec(F(7));
        std::vector<pt::Palm> palms;
        dec.decode(reinterpret_cast<const float*>(raw.data()), lb, palms);
        std::FILE* f = std::fopen(argv[8], "w");
        for (const auto& p : palms) {
            std::fprintf(f, "%.9g %.9g %.9g %.9g %.9g", p.x1, p.y1, p.x2, p.y2, p.score);
            for (float v : p.pts) std::fprintf(f, " %.9g", v);
            const pt::RotRect r = pt::hand_roi_from_palm(p);
            std::fprintf(f, " %.9g %.9g %.9g %.9g\n", r.cx, r.cy, r.size, r.angle);
        }
        std::fclose(f);
        return 0;
    }
    if (cmd == "hand" && argc >= 8) {
        const auto raw = read_file(argv[2]);
        const pt::RotRect r{F(3), F(4), F(5), F(6)};
        pt::HandLandmarks h;
        pt::decode_hand(reinterpret_cast<const float*>(raw.data()), r, 224, h);
        std::FILE* f = std::fopen(argv[7], "w");
        for (int i = 0; i < 21; ++i) std::fprintf(f, "%.9g %.9g %.9g\n", h.x[i], h.y[i], h.z[i]);
        std::fprintf(f, "presence %.9g handedness %.9g\n", h.presence, h.handedness);
        const pt::RotRect n = pt::hand_roi_from_landmarks(h);
        std::fprintf(f, "next %.9g %.9g %.9g %.9g\n", n.cx, n.cy, n.size, n.angle);
        print_shape(f, h);
        std::fclose(f);
        return 0;
    }
    if (cmd == "gesture" && argc >= 3) {
        std::FILE* f = std::fopen(argv[2], "r");
        pt::HandLandmarks h{};
        for (int i = 0; i < 21; ++i)
            if (std::fscanf(f, "%f %f %f %f %f", &h.x[i], &h.y[i], &h.wx[i], &h.wy[i], &h.wz[i]) != 5) return 2;
        std::fclose(f);
        print_shape(stdout, h);
        return 0;
    }
    std::fprintf(stderr, "usage: see the comment at the top of %s\n", __FILE__);
    return 2;
}
