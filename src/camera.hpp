// Frame source on its own thread, decoding straight into pinned (page-locked) memory so the GPU copy
// needs no extra CPU copy.
//
// Sources:
//   videohub        Unitree's head-camera stream, the lab's tested pipeline (BGR via videoconvert)
//   videohub-nv12   same stream, NV12 straight from the hardware decoder (no CPU colour conversion;
//                   the GPU letterbox kernel converts)
//   <file>.h264     raw H.264 recorded by scripts/record_clip.sh, decoded by the hardware decoder
//   <file>.mp4 ...  any video file OpenCV can read (CPU decode)
//   /dev/videoN     V4L2 camera (Option B: only works if videohub_pc4 has been stopped -- ask first)
//   "... ! appsink" any GStreamer pipeline ending in appsink (BGR or NV12)
// Live sources keep only the newest frame (older ones are dropped and counted); files deliver every frame.
// Hardware resize (out_w/out_h, videohub and .h264 sources): the Jetson's VIC (nvvidconv) scales the decoded
// frame to the model's picture size and hands over NV12, so the CPU never touches a full-size frame.
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "preprocess_math.hpp"

namespace pt {

struct CameraConfig {
    std::string source = "videohub";
    std::string format = "auto";  // for .h264 files: bgr | nv12 (auto = nv12)
    std::string dec_opts;         // extra nvv4l2decoder properties, e.g. "enable-max-performance=1"
    int width = 1280, height = 720, fps = 15;  // only for /dev/videoN
    int out_w = 0, out_h = 0;  // > 0: VIC scales frames to this size (NV12; videohub / .h264 sources)
    int every = 1;             // deliver source frames 1, 1+N, 1+2N, ... (the others are skipped uncopied)
};

struct Frame {
    const uint8_t* data = nullptr;  // pinned host memory (GPU-copyable)
    size_t bytes = 0;
    int w = 0, h = 0, stride = 0;   // stride in bytes (NV12: of each plane)
    PixelFormat fmt = PixelFormat::BGR;
    uint64_t seq = 0;               // 1, 2, 3 ... as read from the source
    double t_ready = 0.0;           // steady clock seconds when the frame was read
    double read_ms = 0.0;           // time inside VideoCapture::read
    int slot = -1;
};

class Camera {
public:
    explicit Camera(const CameraConfig& cfg);
    ~Camera();
    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;

    // Waits up to timeout_ms for a frame: 1 = got one, 0 = timed out, -1 = source ended / stopped.
    int next(Frame& f, int timeout_ms);
    void release(const Frame& f);  // hand the frame's memory back
    void stop();

    int width() const { return w_; }
    int height() const { return h_; }
    PixelFormat format() const { return fmt_; }
    bool live() const { return live_; }
    double source_fps() const { return src_fps_; }  // from the file / stream caps (0 if unknown)
    double capture_fps() const { return capture_fps_.load(); }
    uint64_t frames_read() const { return frames_read_.load(); }
    uint64_t frames_dropped() const { return frames_dropped_.load(); }
    uint64_t frames_skipped() const { return frames_skipped_.load(); }  // by `every`
    bool hw_resized() const { return hw_resized_; }
    const std::string& description() const { return desc_; }

private:
    void loop();
    int take_free_slot_locked();

    cv::VideoCapture* cap_ = nullptr;  // heap: if the reader thread is stuck we must not destroy it
    std::string desc_;
    bool live_ = true;
    bool hw_resized_ = false;
    int every_ = 1;
    int w_ = 0, h_ = 0, stride_ = 0, rows_ = 0, type_ = 0;
    PixelFormat fmt_ = PixelFormat::BGR;
    double src_fps_ = 0.0;
    size_t bytes_ = 0;

    struct Slot {
        uint8_t* data = nullptr;
        enum State { Free, Writing, Ready, InUse } state = Free;
        Frame frame;
    };
    std::vector<Slot> slots_;
    std::deque<int> ready_;  // oldest first
    std::mutex m_;
    std::condition_variable cv_;
    std::thread thread_;
    std::atomic<bool> running_{true};
    std::atomic<bool> finished_{false};
    std::atomic<double> capture_fps_{0.0};
    std::atomic<uint64_t> frames_read_{0}, frames_dropped_{0}, frames_skipped_{0};
};

}  // namespace pt
