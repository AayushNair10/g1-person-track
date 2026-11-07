// A TensorRT 8.5 engine (built by trtexec on the robot) with one float input (1x3xHxW or 1xHxWx3) and one
// float output.
// Uses the TensorRT 8.5 binding API (getBindingName, enqueueV2). The inference can be recorded once as
// a CUDA graph and replayed every frame, which removes most kernel-launch work from the CPU.
// Engines built for a DLA core (trtexec --useDLACore=N --allowGPUFallback) need dla_core = N.
#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <string>

namespace pt {

struct Tensor {
    std::string name;
    nvinfer1::Dims dims{};
    size_t count = 0;        // number of floats
    float* dev = nullptr;    // GPU buffer (owned)
    float* host = nullptr;   // pinned CPU buffer of the same size (owned; outputs only)
};

class TrtEngine {
public:
    TrtEngine(const std::string& engine_path, cudaStream_t stream, bool use_cuda_graph, int dla_core = -1);
    ~TrtEngine();
    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    const Tensor& input() const { return in_; }
    const Tensor& output() const { return out_; }
    bool input_nhwc() const { return nhwc_; }
    int input_w() const { return nhwc_ ? in_.dims.d[2] : in_.dims.d[3]; }
    int input_h() const { return in_.dims.d[1 + (nhwc_ ? 0 : 1)]; }

    void enqueue();   // run on the stream (async)
    void download();  // output GPU -> its pinned host buffer (async)
    void download(float* host);  // output GPU -> another pinned host buffer of output().count floats (async)
    bool graph_active() const { return graph_exec_ != nullptr; }
    const std::string& path() const { return path_; }

private:
    std::string path_;
    cudaStream_t stream_;
    bool use_graph_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    Tensor in_, out_;
    void* bindings_[2] = {nullptr, nullptr};
    bool nhwc_ = false;
    int runs_ = 0;
    bool graph_failed_ = false;
    cudaGraphExec_t graph_exec_ = nullptr;
};

}  // namespace pt
