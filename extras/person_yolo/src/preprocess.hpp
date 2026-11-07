// GPU letterbox: camera frame (BGR or NV12, 8-bit) -> model input (RGB planar float, 0..1, grey 114
// padding) in one CUDA kernel, written straight into the TensorRT input buffer.
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

#include "preprocess_math.hpp"

namespace pt {

// src: GPU-readable frame (device memory, or mapped pinned host memory). stride: bytes per row
// (NV12: of the Y plane; the UV plane follows right after src_h rows). Asynchronous on `stream`.
void letterbox_gpu(const uint8_t* src, int stride, PixelFormat fmt, const Letterbox& lb, float* dst,
                   cudaStream_t stream);

}  // namespace pt
