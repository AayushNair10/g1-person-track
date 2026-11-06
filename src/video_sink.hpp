// Annotated MP4 writer on its own thread (so encoding never slows the tracking loop).
// Stops by itself after `seconds` of video; frames are copied only while recording.
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

#include "ocsort.hpp"
#include "preprocess_math.hpp"

namespace pt {

struct Overlay {
    uint64_t frame = 0;
    std::vector<TrackOut> tracks;
    bool has_target = false;
    TrackOut target{};
    double fps = 0, latency_ms = 0;
};

class VideoSink {
public:
    VideoSink(const std::string& path, int w, int h, PixelFormat fmt, double fps, double seconds);
    ~VideoSink();  // writes what is queued, closes the file
    VideoSink(const VideoSink&) = delete;
    VideoSink& operator=(const VideoSink&) = delete;

    bool ok() const { return ok_; }
    bool wants_more() const { return ok_ && pushed_ < max_frames_; }
    void push(const uint8_t* data, size_t bytes, const Overlay& ov);
    int written() const { return written_.load(); }
    int dropped() const { return dropped_; }
    const std::string& path() const { return path_; }

private:
    void loop();
    void draw(cv::Mat& bgr, const Overlay& ov) const;

    std::string path_;
    int w_, h_;
    PixelFormat fmt_;
    bool ok_ = false;
    int max_frames_ = 0, pushed_ = 0, dropped_ = 0;
    cv::VideoWriter writer_;
    struct Job {
        std::vector<uint8_t> data;
        Overlay ov;
    };
    std::deque<Job> q_;
    std::mutex m_;
    std::condition_variable cv_;
    bool done_ = false;
    std::atomic<int> written_{0};
    std::thread thread_;
};

}  // namespace pt
