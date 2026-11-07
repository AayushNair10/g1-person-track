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

#include "preprocess_math.hpp"

namespace pt {

// What to draw on a frame (camera pixels; scaled with box_scale when the frame was resized).
struct OverlayBox {
    float x1, y1, x2, y2;
    bool highlight;     // the followed person's face: green, thick
    std::string label;
};
struct OverlayHand {
    float x[21], y[21];  // landmarks
    bool highlight;
    std::string label;   // e.g. "right: Thumbs up"
};
struct Overlay {
    uint64_t frame = 0;
    double fps = 0, latency_ms = 0;
    std::vector<OverlayBox> boxes;
    std::vector<float> points;  // x, y pairs (face points)
    std::vector<OverlayHand> hands;
    std::string status;         // appended to the status line
};

// Camera frame (BGR or NV12, as the camera delivers it) -> BGR image.
void frame_to_bgr(const uint8_t* data, int w, int h, PixelFormat fmt, cv::Mat& bgr);
// Draws the tracks (grey), the target (green) and a status line. box_scale: frame pixels per camera pixel.
void draw_overlay(cv::Mat& img, const Overlay& ov, float box_scale);

class VideoSink {
public:
    // box_scale: frame pixels per camera pixel (boxes are in camera pixels; < 1 after a hardware resize)
    VideoSink(const std::string& path, int w, int h, PixelFormat fmt, double fps, double seconds,
              float box_scale = 1.f);
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

    std::string path_;
    int w_, h_;
    PixelFormat fmt_;
    float box_scale_;
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
