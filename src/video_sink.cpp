#include "video_sink.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdio>

namespace pt {

VideoSink::VideoSink(const std::string& path, int w, int h, PixelFormat fmt, double fps, double seconds,
                     float box_scale)
    : path_(path), w_(w), h_(h), fmt_(fmt), box_scale_(box_scale) {
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
        frame_to_bgr(job.data.data(), w_, h_, fmt_, bgr);
        draw_overlay(bgr, job.ov, box_scale_);
        writer_.write(bgr);
        ++written_;
    }
}

void frame_to_bgr(const uint8_t* data, int w, int h, PixelFormat fmt, cv::Mat& bgr) {
    if (fmt == PixelFormat::NV12)
        cv::cvtColor(cv::Mat(h * 3 / 2, w, CV_8UC1, const_cast<uint8_t*>(data)), bgr, cv::COLOR_YUV2BGR_NV12);
    else
        cv::Mat(h, w, CV_8UC3, const_cast<uint8_t*>(data)).copyTo(bgr);
}

void draw_overlay(cv::Mat& img, const Overlay& ov, float box_scale) {
    static const int kBones[21][2] = {{0, 1},   {1, 2},   {2, 3},   {3, 4},   {0, 5},   {5, 6},   {6, 7},
                                      {7, 8},   {5, 9},   {9, 10},  {10, 11}, {11, 12}, {9, 13},  {13, 14},
                                      {14, 15}, {15, 16}, {13, 17}, {0, 17},  {17, 18}, {18, 19}, {19, 20}};
    const double scale = std::max(0.5, img.cols / 1280.0);
    const int thin = std::max(1, static_cast<int>(2 * scale)), thick = std::max(2, static_cast<int>(4 * scale));
    auto P = [&](float x, float y) { return cv::Point(static_cast<int>(x * box_scale), static_cast<int>(y * box_scale)); };
    const cv::Scalar green(0, 220, 0), grey(160, 160, 160), orange(0, 160, 255), yellow(0, 255, 255);
    for (const auto& b : ov.boxes) {
        const cv::Scalar col = b.highlight ? green : grey;
        cv::rectangle(img, P(b.x1, b.y1), P(b.x2, b.y2), col, b.highlight ? thick : thin);
        if (!b.label.empty())
            cv::putText(img, b.label, cv::Point(P(b.x1, b.y1).x + 2, std::max(P(b.x1, b.y1).y - 6, 14)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6 * scale, col, thin);
    }
    for (size_t i = 0; i + 1 < ov.points.size(); i += 2)
        cv::circle(img, P(ov.points[i], ov.points[i + 1]), std::max(2, thin), yellow, cv::FILLED);
    for (const auto& h : ov.hands) {
        const cv::Scalar col = h.highlight ? orange : grey;
        for (const auto& b : kBones) cv::line(img, P(h.x[b[0]], h.y[b[0]]), P(h.x[b[1]], h.y[b[1]]), col, thin);
        for (int i = 0; i < 21; ++i) cv::circle(img, P(h.x[i], h.y[i]), std::max(2, thin), col, cv::FILLED);
        if (!h.label.empty()) {
            const float top = *std::min_element(h.y, h.y + 21), left = *std::min_element(h.x, h.x + 21);
            cv::putText(img, h.label, cv::Point(P(left, top).x, std::max(P(left, top).y - 8, 14)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7 * scale, col, thick);
        }
    }
    char txt[160];
    std::snprintf(txt, sizeof(txt), "frame %llu  %.1f FPS  latency %.1f ms  %s",
                  static_cast<unsigned long long>(ov.frame), ov.fps, ov.latency_ms, ov.status.c_str());
    cv::putText(img, txt, cv::Point(10, static_cast<int>(30 * scale)), cv::FONT_HERSHEY_SIMPLEX, 0.8 * scale, yellow,
                thin);
}

}  // namespace pt
