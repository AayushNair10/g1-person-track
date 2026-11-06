// Live view in a browser: open http://<robot>:<port>/ on the laptop (same idea as FaceHandTrack's viewer).
//   /             small page showing the stream
//   /stream       MJPEG stream (multipart/x-mixed-replace)
//   /snapshot.jpg the newest frame
// The tracking loop only hands over a copy of the frame, and only while someone is watching and at most
// max_fps times a second; colour conversion, drawing and JPEG encoding happen on the view's own thread, so
// the tracker never waits for the view. With nobody watching it costs nothing but an idle thread.
// Uses only POSIX sockets and OpenCV (imgcodecs, imgproc), both already on the robot.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "video_sink.hpp"

namespace pt {

class LiveView {
public:
    // box_scale: frame pixels per camera pixel (boxes are in camera pixels). If the port cannot be opened the
    // view is off (ok() == false) and the program carries on without it.
    LiveView(int port, double max_fps, int frame_w, int frame_h, PixelFormat fmt, float box_scale,
             int jpeg_quality = 80);
    ~LiveView();
    LiveView(const LiveView&) = delete;
    LiveView& operator=(const LiveView&) = delete;

    bool ok() const { return listen_fd_ >= 0; }
    int port() const { return port_; }
    bool has_viewers() const { return viewers_.load() > 0; }
    // true when a frame should be handed over now (someone watching, rate limit reached)
    bool wants_frame(double now_s) const;
    // Copies the frame; the newest one replaces any not yet encoded.
    void submit(const uint8_t* data, size_t bytes, const Overlay& ov, double now_s);
    uint64_t frames_sent() const { return encoded_.load(); }

private:
    void accept_loop();
    void encode_loop();
    void serve(int fd);

    int port_;
    double min_interval_;
    int w_, h_;
    PixelFormat fmt_;
    float box_scale_;
    int jpeg_quality_;
    int listen_fd_ = -1;
    std::atomic<bool> running_{true};
    std::atomic<int> viewers_{0}, connections_{0};
    std::atomic<double> last_submit_{-1e9};
    std::atomic<uint64_t> encoded_{0};
    std::thread accept_thread_, encode_thread_;

    std::mutex in_m_;  // frame waiting to be encoded
    std::condition_variable in_cv_;
    std::vector<uint8_t> pending_;
    Overlay pending_ov_;
    bool has_pending_ = false;

    std::mutex out_m_;  // newest JPEG
    std::condition_variable out_cv_;
    std::vector<unsigned char> jpeg_;
    uint64_t jpeg_id_ = 0;

    std::mutex conn_m_;  // open client sockets, closed on shutdown
    std::set<int> clients_;
};

}  // namespace pt
