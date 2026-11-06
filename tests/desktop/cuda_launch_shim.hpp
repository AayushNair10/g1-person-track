// Parse-only helper for tests/desktop/syntax_check.sh: lets clang's CUDA mode (-x cuda, no real CUDA
// install) accept kernel launches. Never used by the real build (nvcc on the robot).
#pragma once
#include <cuda_runtime_api.h>
extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, cudaStream_t stream = 0);
