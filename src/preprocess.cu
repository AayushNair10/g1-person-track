// CUDA side of preprocess.hpp (built by nvcc, see Makefile). One thread per model-input pixel; the arithmetic
// is preprocess_math.hpp, the same code the desktop test runs on the CPU.
#include "cuda_utils.hpp"
#include "preprocess.hpp"

namespace pt {

namespace {

__global__ void letterbox_kernel(GpuFrame f, Letterbox lb, InputFormat in, float* dst) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x < lb.dst_w && y < lb.dst_h) letterbox_pixel(f.data, f.stride, f.fmt, f.h, lb, in, dst, x, y);
}

__global__ void rotcrop_kernel(GpuFrame f, RotRect r, int out, InputFormat in, float* dst) {
    const int u = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int v = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (u < out && v < out) rotcrop_pixel(f.data, f.stride, f.fmt, f.w, f.h, r, out, in, dst, u, v);
}

}  // namespace

void letterbox_gpu(const GpuFrame& f, const Letterbox& lb, const InputFormat& in, float* dst, cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid((lb.dst_w + block.x - 1) / block.x, (lb.dst_h + block.y - 1) / block.y);
    letterbox_kernel<<<grid, block, 0, stream>>>(f, lb, in, dst);
    CUDA_CHECK(cudaGetLastError());
}

void rotcrop_gpu(const GpuFrame& f, const RotRect& r, int out, const InputFormat& in, float* dst, cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid((out + block.x - 1) / block.x, (out + block.y - 1) / block.y);
    rotcrop_kernel<<<grid, block, 0, stream>>>(f, r, out, in, dst);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace pt
