// Desktop check of the robot's input and output code on the CPU (the CUDA kernel runs the same
// letterbox_pixel() per pixel). Used by check_pipeline.py.
//   test_pipeline pre  <frame.raw> <w> <h> <bgr|nv12> <dst_w> <dst_h> <tensor_out.f32>
//   test_pipeline post <output.f32> <d0> <d1> <d2> <src_w> <src_h> <dst_w> <dst_h> <conf> <dets_out.txt>
//                      [<frame_w> <frame_h>]   frame given to the letterbox was hardware-resized from src_w x src_h
//                                              (main.cpp's --hw-resize mapping: for_camera)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../src/postprocess.hpp"
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

int main(int argc, char** argv) {
    if (argc >= 9 && std::strcmp(argv[1], "pre") == 0) {
        const auto src = read_file(argv[2]);
        const int w = std::atoi(argv[3]), h = std::atoi(argv[4]);
        const pt::PixelFormat fmt = std::string(argv[5]) == "nv12" ? pt::PixelFormat::NV12 : pt::PixelFormat::BGR;
        const int dw = std::atoi(argv[6]), dh = std::atoi(argv[7]);
        const pt::Letterbox lb = pt::make_letterbox(w, h, dw, dh);
        const int stride = fmt == pt::PixelFormat::BGR ? w * 3 : w;
        std::vector<float> dst(static_cast<size_t>(3) * dw * dh);
        for (int y = 0; y < dh; ++y)
            for (int x = 0; x < dw; ++x) pt::letterbox_pixel(src.data(), stride, fmt, lb, dst.data(), x, y);
        std::FILE* f = std::fopen(argv[8], "wb");
        std::fwrite(dst.data(), sizeof(float), dst.size(), f);
        std::fclose(f);
        std::printf("letterbox new %dx%d left %d top %d\n", lb.new_w, lb.new_h, lb.left, lb.top);
        return 0;
    }
    if (argc >= 12 && std::strcmp(argv[1], "post") == 0) {
        const auto raw = read_file(argv[2]);
        const int dims[3] = {std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5])};
        pt::OutputLayout layout;
        if (!pt::layout_from_dims(dims, 3, layout)) {
            std::fprintf(stderr, "unknown output layout\n");
            return 2;
        }
        const int src_w = std::atoi(argv[6]), src_h = std::atoi(argv[7]), dst_w = std::atoi(argv[8]),
                  dst_h = std::atoi(argv[9]);
        const pt::Letterbox lb =
            argc >= 14 ? pt::for_camera(pt::make_letterbox(std::atoi(argv[12]), std::atoi(argv[13]), dst_w, dst_h),
                                        src_w, src_h)
                       : pt::make_letterbox(src_w, src_h, dst_w, dst_h);
        pt::Decoder dec(layout, lb, static_cast<float>(std::atof(argv[10])));
        std::vector<pt::Det> dets;
        dec.decode(reinterpret_cast<const float*>(raw.data()), dets);
        std::FILE* f = std::fopen(argv[11], "w");
        for (const auto& d : dets) std::fprintf(f, "%.9g %.9g %.9g %.9g %.9g\n", d.x1, d.y1, d.x2, d.y2, d.score);
        std::fclose(f);
        return 0;
    }
    std::fprintf(stderr, "usage: see the comment at the top of %s\n", __FILE__);
    return 2;
}
