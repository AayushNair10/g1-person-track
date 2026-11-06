// Small CUDA helpers.
#pragma once

#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>

namespace pt {

inline void cuda_check(cudaError_t e, const char* what, const char* file, int line) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[cuda] %s failed (%s:%d): %s\n", what, file, line, cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace pt

#define CUDA_CHECK(x) ::pt::cuda_check((x), #x, __FILE__, __LINE__)
