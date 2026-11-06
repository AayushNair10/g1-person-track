#include "trt_engine.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <vector>

#include "cuda_utils.hpp"

namespace pt {

namespace {

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::fprintf(stderr, "[trt] %s\n", msg);
    }
};

Logger& logger() {
    static Logger l;
    return l;
}

[[noreturn]] void die(const std::string& path, const char* what) {
    std::fprintf(stderr, "[trt] %s: %s\n", path.c_str(), what);
    std::exit(1);
}

}  // namespace

TrtEngine::TrtEngine(const std::string& engine_path, cudaStream_t stream, bool use_cuda_graph, int dla_core)
    : path_(engine_path), stream_(stream), use_graph_(use_cuda_graph) {
    std::ifstream f(engine_path, std::ios::binary);
    if (!f) die(path_, "cannot open the engine file (build it with scripts/build_engines.sh)");
    const std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    runtime_ = nvinfer1::createInferRuntime(logger());
    if (!runtime_) die(path_, "createInferRuntime failed");
    if (dla_core >= 0) {
        if (dla_core >= runtime_->getNbDLACores()) die(path_, "this device has no such DLA core");
        runtime_->setDLACore(dla_core);
    }
    engine_ = runtime_->deserializeCudaEngine(blob.data(), blob.size());
    if (!engine_) die(path_, "cannot deserialize (engine built by another TensorRT version or device?)");
    context_ = engine_->createExecutionContext();
    if (!context_) die(path_, "createExecutionContext failed");

    if (engine_->getNbBindings() != 2) die(path_, "expected exactly one input and one output");
    for (int i = 0; i < 2; ++i) {
        if (engine_->getBindingDataType(i) != nvinfer1::DataType::kFLOAT)
            die(path_, "input/output must be float32 (build with trtexec defaults: --fp16 keeps float I/O)");
        Tensor& t = engine_->bindingIsInput(i) ? in_ : out_;
        if (!t.name.empty()) die(path_, "expected one input and one output");
        t.name = engine_->getBindingName(i);
        t.dims = engine_->getBindingDimensions(i);
        t.count = 1;
        for (int k = 0; k < t.dims.nbDims; ++k) {
            if (t.dims.d[k] <= 0) die(path_, "dynamic shapes are not supported (export with dynamic=False)");
            t.count *= static_cast<size_t>(t.dims.d[k]);
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&t.dev), t.count * sizeof(float)));
        CUDA_CHECK(cudaMemset(t.dev, 0, t.count * sizeof(float)));
        bindings_[i] = t.dev;
    }
    if (in_.dims.nbDims != 4 || in_.dims.d[0] != 1 || in_.dims.d[1] != 3)
        die(path_, "input must be 1x3xHxW");
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&out_.host), out_.count * sizeof(float)));
}

TrtEngine::~TrtEngine() {
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    cudaFree(in_.dev);
    cudaFree(out_.dev);
    cudaFreeHost(out_.host);
    delete context_;
    delete engine_;
    delete runtime_;
}

void TrtEngine::enqueue() {
    if (graph_exec_) {
        CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream_));
        return;
    }
    // First run: plain (TensorRT finishes its lazy set-up). Second run: record it as a CUDA graph.
    if (use_graph_ && !graph_failed_ && runs_ >= 1) {
        cudaGraph_t graph = nullptr;
        bool ok = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
        if (ok) {
            const bool queued = context_->enqueueV2(bindings_, stream_, nullptr);
            ok = cudaStreamEndCapture(stream_, &graph) == cudaSuccess && queued && graph != nullptr;
        }
        if (ok) ok = cudaGraphInstantiate(&graph_exec_, graph, nullptr, nullptr, 0) == cudaSuccess;
        if (graph) cudaGraphDestroy(graph);
        if (ok) {
            CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream_));
            ++runs_;
            return;
        }
        cudaGetLastError();  // clear the failed capture
        graph_exec_ = nullptr;
        graph_failed_ = true;
        std::fprintf(stderr, "[trt] %s: CUDA graph recording failed, running without it\n", path_.c_str());
    }
    if (!context_->enqueueV2(bindings_, stream_, nullptr)) die(path_, "enqueueV2 failed");
    ++runs_;
}

void TrtEngine::download() {
    CUDA_CHECK(cudaMemcpyAsync(out_.host, out_.dev, out_.count * sizeof(float), cudaMemcpyDeviceToHost, stream_));
}

}  // namespace pt
