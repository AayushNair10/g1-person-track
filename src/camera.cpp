#include "camera.hpp"

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "cuda_utils.hpp"
#include "timing.hpp"

namespace pt {

namespace {

// Unitree's videohub_pc4 owns the head camera (RealSense D435i colour, 1920x1080 @ 15 fps) and multicasts
// it as 1280x720 H.264 RTP on eth0. Receiving that stream keeps Unitree's own video working; the Jetson's
// hardware decoder (nvv4l2decoder) unpacks it. This prefix is the lab's tested pipeline up to the decoder.
const char* kVideohubPrefix =
    "udpsrc multicast-group=230.1.1.1 port=1720 multicast-iface=eth0 auto-multicast=true "
    "caps=\"application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96\" ! "
    "rtph264depay ! h264parse ! nvv4l2decoder";

// After the decoder. BGR = the lab's exact tail (videoconvert does BGRx -> BGR on the CPU).
// NV12 = the decoder's own format, copied out by nvvidconv (VIC hardware), no CPU conversion.
// Scaled (out_w > 0) = NV12 resized by the VIC to out_w x out_h on the way out (no extra pass);
// `interp` = nvvidconv interpolation-method (1 = bilinear; empty = nvvidconv's default filter).
std::string tail(bool nv12, bool live, int out_w, int out_h, const std::string& interp) {
    const std::string sink = live ? "appsink drop=true max-buffers=1 sync=false"
                                  : "appsink drop=false max-buffers=4 sync=false";
    if (out_w > 0) {
        const std::string conv = interp.empty() ? " ! nvvidconv" : " ! nvvidconv interpolation-method=" + interp;
        return conv + " ! video/x-raw,format=NV12,width=" + std::to_string(out_w) + ",height=" +
               std::to_string(out_h) + " ! " + sink;
    }
    if (nv12) return " ! nvvidconv ! video/x-raw,format=NV12 ! " + sink;
    return " ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! " + sink;
}

bool ends_with(const std::string& s, const char* suffix) {
    const size_t n = std::strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "[camera] %s\n", msg.c_str());
    std::exit(1);
}

}  // namespace

bool probe_frame_size(const CameraConfig& cfg, int& w, int& h) {
    cv::VideoCapture cap;
    const std::string& src = cfg.source;
    const std::string dec = cfg.dec_opts.empty() ? "" : " " + cfg.dec_opts;
    bool ok;
    if (ends_with(src, ".h264") || ends_with(src, ".264"))
        ok = cap.open("filesrc location=\"" + src + "\" ! h264parse ! nvv4l2decoder" + dec + tail(true, false, 0, 0, ""),
                      cv::CAP_GSTREAMER);
    else
        ok = cap.open(src);
    cv::Mat m;
    if (!ok || !cap.read(m) || m.empty()) return false;
    w = m.cols;
    h = m.type() == CV_8UC1 && m.rows % 3 == 0 ? m.rows * 2 / 3 : m.rows;
    return true;
}

Camera::Camera(const CameraConfig& cfg) : cap_(new cv::VideoCapture()), every_(cfg.every > 1 ? cfg.every : 1) {
    const std::string& src = cfg.source;
    const std::string dec = cfg.dec_opts.empty() ? "" : " " + cfg.dec_opts;
    const bool scaled = cfg.out_w > 0 && cfg.out_h > 0;
    const std::string scaled_desc =
        scaled ? ", VIC-scaled to " + std::to_string(cfg.out_w) + "x" + std::to_string(cfg.out_h) : "";
    // GStreamer source ending in the hardware decoder, then one of the tails above
    auto open_gst = [&](const std::string& head, bool nv12, bool live) {
        std::string pipe = head + tail(nv12, live, scaled ? cfg.out_w : 0, cfg.out_h, "1");
        std::fprintf(stderr, "[camera] pipeline: %s\n", pipe.c_str());
        bool opened = cap_->open(pipe, cv::CAP_GSTREAMER);
        if (!opened && scaled) {  // nvvidconv without interpolation-method: its default filter
            pipe = head + tail(nv12, live, cfg.out_w, cfg.out_h, "");
            std::fprintf(stderr, "[camera] retrying with nvvidconv's default filter: %s\n", pipe.c_str());
            opened = cap_->open(pipe, cv::CAP_GSTREAMER);
        }
        hw_resized_ = opened && scaled;
        return opened;
    };
    bool ok = false;
    if (src == "videohub" || src == "videohub-bgr" || src == "videohub-nv12") {
        const bool nv12 = src == "videohub-nv12" || scaled;
        desc_ = std::string("videohub stream rtp://230.1.1.1:1720 (") + (nv12 ? "NV12" : "BGR") + scaled_desc + ")";
        live_ = true;
        ok = open_gst(kVideohubPrefix + dec, nv12, true);
    } else if (ends_with(src, ".h264") || ends_with(src, ".264")) {
        const bool nv12 = cfg.format != "bgr" || scaled;
        desc_ = "H.264 file " + src + (nv12 ? " (NV12" : " (BGR") + scaled_desc + ")";
        live_ = false;
        ok = open_gst("filesrc location=\"" + src + "\" ! h264parse ! nvv4l2decoder" + dec, nv12, false);
    } else if (src.rfind("/dev/video", 0) == 0) {
        // Option B: the camera directly. Only possible after videohub_pc4 has been stopped by hand -- this
        // program never stops it. Two ways in:
        //   default        OpenCV V4L2, device opened by index, YUYV, OpenCV converts to BGR on the CPU
        //                  (the lab guide's method)
        //   --format nv12  GStreamer v4l2src, YUYV converted to NV12 by the VIC (no CPU colour conversion);
        //   / --hw-resize  the VIC also scales; falls back to the default way if the pipeline does not start
        live_ = true;
        const std::string mode = std::to_string(cfg.width) + "x" + std::to_string(cfg.height) + " @ " +
                                 std::to_string(cfg.fps) + " FPS";
        if (cfg.format == "nv12" || scaled) {
            desc_ = "camera " + src + " " + mode + " (GStreamer, YUYV -> NV12 on the VIC" + scaled_desc + ")";
            ok = open_gst("v4l2src device=" + src + " ! video/x-raw,format=YUY2,width=" + std::to_string(cfg.width) +
                              ",height=" + std::to_string(cfg.height) + ",framerate=" + std::to_string(cfg.fps) + "/1",
                          true, true);
            if (!ok) std::fprintf(stderr, "[camera] GStreamer v4l2src did not start; trying OpenCV V4L2 (BGR)\n");
        }
        if (!ok) {
            desc_ = "camera " + src + " " + mode + " (OpenCV V4L2, YUYV -> BGR on the CPU)";
            if (src.size() > 10) ok = cap_->open(std::atoi(src.c_str() + 10), cv::CAP_V4L2);  // by index: more
            if (!ok) ok = cap_->open(src, cv::CAP_V4L2);                                         // reliable in 4.2
            if (ok) {  // format first, then size, then rate
                cap_->set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
                cap_->set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
                cap_->set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);
                cap_->set(cv::CAP_PROP_FPS, cfg.fps);
                cap_->set(cv::CAP_PROP_BUFFERSIZE, 1);
                const int fcc = static_cast<int>(cap_->get(cv::CAP_PROP_FOURCC));
                const char f4[5] = {static_cast<char>(fcc & 255), static_cast<char>((fcc >> 8) & 255),
                                    static_cast<char>((fcc >> 16) & 255), static_cast<char>((fcc >> 24) & 255), 0};
                std::fprintf(stderr, "[camera] %s: driver reports %.0fx%.0f @ %.1f FPS, format %s\n", src.c_str(),
                             cap_->get(cv::CAP_PROP_FRAME_WIDTH), cap_->get(cv::CAP_PROP_FRAME_HEIGHT),
                             cap_->get(cv::CAP_PROP_FPS), f4);
            }
        }
    } else if (src.find('!') != std::string::npos) {
        desc_ = "GStreamer pipeline";
        live_ = src.find("filesrc") == std::string::npos;
        ok = cap_->open(src, cv::CAP_GSTREAMER);
    } else {
        desc_ = "video file " + src;
        live_ = false;
        ok = cap_->open(src);
    }
    if (scaled && !hw_resized_ && ok)
        std::fprintf(stderr, "[camera] hardware resize only works for videohub / .h264 / /dev/videoN (GStreamer); "
                             "full-size frames\n");
    if (!ok)
        die("cannot open " + desc_ +
            "\n  videohub: is videohub_pc4 running, and was OpenCV built with GStreamer?"
            "\n  /dev/videoN: is videohub_pc4 still holding it (sudo fuser -v /dev/video*)? is this a size / rate"
            "\n  the camera offers (v4l2-ctl -d /dev/video4 --list-formats-ext)? permission (ls -l /dev/video4)?");

    cv::Mat probe;
    if (!cap_->read(probe) || probe.empty()) die(desc_ + " opened but gives no frames");
    if (!probe.isContinuous()) probe = probe.clone();
    if (probe.type() == CV_8UC3) {
        fmt_ = PixelFormat::BGR;
        w_ = probe.cols;
        h_ = probe.rows;
        stride_ = probe.cols * 3;
    } else if (probe.type() == CV_8UC1 && probe.rows % 3 == 0) {
        fmt_ = PixelFormat::NV12;  // OpenCV hands NV12 over as one (h * 3/2) x w plane
        w_ = probe.cols;
        h_ = probe.rows * 2 / 3;
        stride_ = probe.cols;
    } else {
        die("unsupported frame type " + std::to_string(probe.type()) + " (" + std::to_string(probe.cols) + "x" +
            std::to_string(probe.rows) + ")");
    }
    rows_ = probe.rows;
    type_ = probe.type();
    bytes_ = probe.total() * probe.elemSize();
    src_fps_ = cap_->get(cv::CAP_PROP_FPS);
    if (!(src_fps_ > 0 && src_fps_ < 1000)) src_fps_ = 0.0;

    slots_.resize(live_ ? 3 : 4);
    for (auto& s : slots_) CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&s.data), bytes_, cudaHostAllocMapped));
    // the probe frame is frame 1
    std::memcpy(slots_[0].data, probe.data, bytes_);
    Frame& f = slots_[0].frame;
    f = Frame{slots_[0].data, bytes_, w_, h_, stride_, fmt_, 1, now_s(), 0.0, 0};
    slots_[0].state = Slot::Ready;
    ready_.push_back(0);
    frames_read_ = 1;

    if (src.rfind("/dev/video", 0) == 0 && !hw_resized_ && (w_ != cfg.width || h_ != cfg.height))
        std::fprintf(stderr, "[camera] asked the camera for %dx%d, got %dx%d (boxes are in the frames' pixels)\n",
                     cfg.width, cfg.height, w_, h_);
    if (hw_resized_ && (w_ != cfg.out_w || h_ != cfg.out_h))
        std::fprintf(stderr, "[camera] asked the VIC for %dx%d, got %dx%d\n", cfg.out_w, cfg.out_h, w_, h_);
    std::fprintf(stderr, "[camera] %s: %dx%d %s, %s%s\n", desc_.c_str(), w_, h_, fmt_ == PixelFormat::NV12 ? "NV12" : "BGR",
                 live_ ? "live (newest frame wins)" : "file (every frame)",
                 every_ > 1 ? (", 1 of every " + std::to_string(every_) + " frames").c_str() : "");
    thread_ = std::thread(&Camera::loop, this);
}

Camera::~Camera() {
    stop();
    if (!finished_) return;  // reader thread still stuck in a read: leave its memory and capture alone
    delete cap_;
    for (auto& s : slots_) cudaFreeHost(s.data);
}

void Camera::stop() {
    running_ = false;
    cv_.notify_all();
    if (!thread_.joinable()) return;
    for (int i = 0; i < 300 && !finished_; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (finished_) {
        thread_.join();
    } else {
        std::fprintf(stderr, "[camera] reader thread did not stop (source stalled); leaving it\n");
        thread_.detach();
    }
}

int Camera::take_free_slot_locked() {
    for (size_t i = 0; i < slots_.size(); ++i)
        if (slots_[i].state == Slot::Free) return static_cast<int>(i);
    return -1;
}

void Camera::loop() {
    double t_win = now_s();
    int n_win = 0;
    uint64_t seq = 1;
    while (running_) {
        // --every N: pull the next N-1 source frames without copying them anywhere
        bool skipped_ok = true;
        for (int k = 1; k < every_ && running_ && skipped_ok; ++k) {
            skipped_ok = cap_->grab();
            if (skipped_ok) {
                ++frames_skipped_;
                ++n_win;
            }
        }
        if (!skipped_ok) {
            if (live_) std::fprintf(stderr, "[camera] stream ended or read failed\n");
            break;
        }
        if (!running_) break;
        int s = -1;
        {
            std::unique_lock<std::mutex> lk(m_);
            if (live_) {
                s = take_free_slot_locked();
                if (s < 0 && !ready_.empty()) {  // cannot happen with 3 slots, but never block a live source
                    s = ready_.front();
                    ready_.pop_front();
                    ++frames_dropped_;
                }
            } else {
                cv_.wait(lk, [&] { return !running_ || take_free_slot_locked() >= 0; });
                if (!running_) break;
                s = take_free_slot_locked();
            }
            if (s < 0) break;
            slots_[s].state = Slot::Writing;
        }
        cv::Mat m(rows_, w_, type_, slots_[s].data);
        const double t0 = now_s();
        const bool ok = cap_->read(m);
        const double t1 = now_s();
        if (!ok || m.empty()) {
            std::lock_guard<std::mutex> lk(m_);
            slots_[s].state = Slot::Free;
            if (live_) std::fprintf(stderr, "[camera] stream ended or read failed\n");
            break;
        }
        if (m.data != slots_[s].data) {  // OpenCV gave its own buffer: copy (same size only)
            if (m.isContinuous() && m.total() * m.elemSize() == bytes_) {
                std::memcpy(slots_[s].data, m.data, bytes_);
            } else {
                std::fprintf(stderr, "[camera] frame size changed mid-stream; stopping\n");
                std::lock_guard<std::mutex> lk(m_);
                slots_[s].state = Slot::Free;
                break;
            }
        }
        ++frames_read_;
        {
            std::lock_guard<std::mutex> lk(m_);
            slots_[s].frame = Frame{slots_[s].data, bytes_, w_, h_, stride_, fmt_, ++seq, t1, 1000.0 * (t1 - t0), s};
            slots_[s].state = Slot::Ready;
            if (live_) {
                while (!ready_.empty()) {  // an unread older frame: drop it, newest wins
                    slots_[ready_.front()].state = Slot::Free;
                    ready_.pop_front();
                    ++frames_dropped_;
                }
            }
            ready_.push_back(s);
        }
        cv_.notify_all();
        ++n_win;
        if (t1 - t_win >= 1.0) {
            capture_fps_ = n_win / (t1 - t_win);
            n_win = 0;
            t_win = t1;
        }
    }
    finished_ = true;
    cv_.notify_all();
}

int Camera::next(Frame& f, int timeout_ms) {
    std::unique_lock<std::mutex> lk(m_);
    cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return !ready_.empty() || finished_ || !running_; });
    if (!running_) return -1;
    if (ready_.empty()) return finished_ ? -1 : 0;
    const int s = ready_.front();
    ready_.pop_front();
    slots_[s].state = Slot::InUse;
    f = slots_[s].frame;
    return 1;
}

void Camera::release(const Frame& f) {
    {
        std::lock_guard<std::mutex> lk(m_);
        if (f.slot >= 0 && f.slot < static_cast<int>(slots_.size())) slots_[f.slot].state = Slot::Free;
    }
    cv_.notify_all();
}

}  // namespace pt
