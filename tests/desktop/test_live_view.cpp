// Desktop check of the live view's HTTP side (src/live_view.cpp): real sockets, threads and shutdown; the
// image work (colour conversion, drawing, JPEG) is stubbed because the Mac has no OpenCV C++ libraries.
// Built and driven by tests/desktop/check_live_view.sh. Prints the port, then serves until stdin closes.
#include <unistd.h>

#include <atomic>
#include <memory>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "../../src/live_view.hpp"

// ---- stubs for the OpenCV / drawing code the view calls (the robot links the real ones)
static std::atomic<int> g_encoded{0};
namespace cv {
bool imencode(const String&, InputArray, std::vector<uchar>& buf, const std::vector<int>&) {
    const std::string s = "FAKEJPEG-" + std::to_string(++g_encoded);
    buf.assign(s.begin(), s.end());
    return true;
}
void Mat::deallocate() {}
void fastFree(void*) {}
void error(int, const String&, const char*, const char*, int) { std::abort(); }
}  // namespace cv
namespace pt {
void frame_to_bgr(const uint8_t*, int, int, PixelFormat, cv::Mat&) {}
void draw_overlay(cv::Mat&, const Overlay&, float) {}
}  // namespace pt

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const int port = argc > 1 ? std::atoi(argv[1]) : 18090;
    {  // a second view on a port in use must switch itself off, not exit
        pt::LiveView a(port, 15, 64, 36, pt::PixelFormat::BGR, 1.f);
        pt::LiveView b(port, 15, 64, 36, pt::PixelFormat::BGR, 1.f);
        std::printf("busy-port: first ok=%d second ok=%d\n", a.ok(), b.ok());
    }
    std::unique_ptr<pt::LiveView> view(new pt::LiveView(port, 15, 64, 36, pt::PixelFormat::BGR, 1.f));
    if (!view->ok()) return 1;
    std::atomic<bool> run{true};
    std::atomic<long> submitted{0};
    std::vector<uint8_t> frame(64 * 36 * 3, 128);
    std::thread tracker([&] {  // the tracking loop: 30 FPS, hands frames over only when asked
        while (run) {
            const double t = now_s();
            if (view->wants_frame(t)) {
                pt::Overlay ov;
                view->submit(frame.data(), frame.size(), ov, t);
                ++submitted;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        }
    });
    std::printf("ready %d\n", port);
    std::fflush(stdout);
    char c;
    while (::read(0, &c, 1) > 0) {  // the driver sends a line to ask for counters; EOF = shut down
        if (c == '\n') {
            std::printf("submitted=%ld encoded=%d viewers=%d\n", submitted.load(), g_encoded.load(),
                        view->has_viewers() ? 1 : 0);
            std::fflush(stdout);
        }
    }
    run = false;
    tracker.join();
    const double t0 = now_s();
    view.reset();  // the driver still has a stream open: shutdown must not hang
    std::printf("view destroyed in %.2f s\n", now_s() - t0);
    return 0;
}
