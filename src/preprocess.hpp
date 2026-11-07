// GPU input preparation: camera frame (BGR or NV12, 8-bit) -> model input (float, the model's layout and
// channel order), one CUDA kernel each, written straight into the TensorRT input buffer. The per-pixel
// arithmetic is in preprocess_math.hpp (also compiled for the CPU desktop tests).
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

#include "preprocess_math.hpp"

namespace pt {

// A frame on the GPU: device memory, or mapped pinned host memory. stride: bytes per row (NV12: of the
// Y plane; the UV plane follows right after h rows).
struct GpuFrame {
    const uint8_t* data = nullptr;
    int stride = 0, w = 0, h = 0;
    PixelFormat fmt = PixelFormat::BGR;
};

// Region of the frame (lb.src_*) resized to fit lb.dst_w x lb.dst_h, centred and padded. Asynchronous.
void letterbox_gpu(const GpuFrame& f, const Letterbox& lb, const InputFormat& in, float* dst, cudaStream_t stream);

// Rotated square r of the frame -> out x out input. Asynchronous.
void rotcrop_gpu(const GpuFrame& f, const RotRect& r, int out, const InputFormat& in, float* dst, cudaStream_t stream);

}  // namespace pt
