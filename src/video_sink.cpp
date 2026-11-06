#include "video_sink.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdio>

namespace pt {

VideoSink::VideoSink(const std::string& path, int w, int h, PixelFormat fmt, double fps, double seconds)
    : path_(path), w_(w), h_(h), fmt_(fmt) {
    if (!(fps > 0)) fps = 15.0;
    max_frames_ = static_cast<int>(seconds * fps + 0.5);
    ok_ = writer_.open(path_, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(w, h));
    if (!ok_) {  // no MPEG-4 encoder: Motion-JPEG AVI always works with OpenCV's own writer
        const size_t dot = path_.rfind('.');
        path_ = (dot == std::string::npos ? path_ : path_.substr(0, dot)) + ".avi";
        ok_ = writer_.open(path_, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(w, h));
    }
    if (!ok_) {
        std::fprintf(stderr, "[video] cannot open a video writer for %s\n", path.c_str());
        return;
    }
    std::fprintf(stderr, "[video] recording %.0f s (%d frames) to %s\n", seconds, max_frames_, path_.c_str());
    thread_ = std::thread(&VideoSink::loop, this);
}

VideoSink::~VideoSink() {
    {
        std::lock_guard<std::mutex> lk(m_);
        done_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    if (ok_) {
        writer_.release();
        std::fprintf(stderr, "[video] wrote %d frames to %s (%d dropped)\n", written_.load(), path_.c_str(), dropped_);
    }
}

void VideoSink::push(const uint8_t* data, size_t bytes, const Overlay& ov) {
    if (!wants_more()) return;
    ++pushed_;
    {
        std::lock_guard<std::mutex> lk(m_);
        if (q_.size() >= 16) {  // encoder behind: drop rather than slow the tracker
            ++dropped_;
            return;
        }
        q_.push_back(Job{std::vector<uint8_t>(data, data + bytes), ov});
    }
    cv_.notify_one();
}

void VideoSink::loop() {
    cv::Mat bgr;
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [&] { return done_ || !q_.empty(); });
            if (q_.empty()) return;
            job = std::move(q_.front());
            q_.pop_front();
        }
        if (fmt_ == PixelFormat::NV12) {
            const cv::Mat yuv(h_ * 3 / 2, w_, CV_8UC1, job.data.data());
            cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12);
        } else {
            cv::Mat(h_, w_, CV_8UC3, job.data.data()).copyTo(bgr);
        }
        draw(bgr, job.ov);
        writer_.write(bgr);
        ++written_;
    }
}

void VideoSink::draw(cv::Mat& img, const Overlay& ov) const {
    const double scale = img.cols / 1280.0;
    const int thin = std::max(1, static_cast<int>(2 * scale)), thick = std::max(2, static_cast<int>(4 * scale));
    char txt[128];
    for (const auto& t : ov.tracks) {
        const bool is_target = ov.has_target && t.id == ov.target.id;
        if (is_target) continue;
        cv::rectangle(img, cv::Point(static_cast<int>(t.x1), static_cast<int>(t.y1)),
                      cv::Point(static_cast<int>(t.x2), static_cast<int>(t.y2)), cv::Scalar(160, 160, 160), thin);
        std::snprintf(txt, sizeof(txt), "id %d", t.id);
        cv::putText(img, txt, cv::Point(static_cast<int>(t.x1) + 4, static_cast<int>(t.y1) + static_cast<int>(22 * scale)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6 * scale, cv::Scalar(160, 160, 160), thin);
    }
    if (ov.has_target) {
        const auto& t = ov.target;
        cv::rectangle(img, cv::Point(static_cast<int>(t.x1), static_cast<int>(t.y1)),
                      cv::Point(static_cast<int>(t.x2), static_cast<int>(t.y2)), cv::Scalar(0, 220, 0), thick);
        std::snprintf(txt, sizeof(txt), "TARGET id %d  %.2f", t.id, t.score);
        cv::putText(img, txt, cv::Point(static_cast<int>(t.x1) + 4, std::max(static_cast<int>(t.y1) - 8, 20)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8 * scale, cv::Scalar(0, 220, 0), thick);
    }
    std::snprintf(txt, sizeof(txt), "frame %llu  %.1f FPS  latency %.1f ms", static_cast<unsigned long long>(ov.frame),
                  ov.fps, ov.latency_ms);
    cv::putText(img, txt, cv::Point(10, static_cast<int>(30 * scale)), cv::FONT_HERSHEY_SIMPLEX, 0.8 * scale,
                cv::Scalar(0, 255, 255), thin);
}

}  // namespace pt
