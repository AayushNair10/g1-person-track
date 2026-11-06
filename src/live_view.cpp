#include "live_view.hpp"

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <opencv2/imgcodecs.hpp>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>

#ifndef MSG_NOSIGNAL  // Linux has it (no SIGPIPE when a browser closes); only for the desktop compile check
#define MSG_NOSIGNAL 0
#endif

namespace pt {

namespace {

bool send_all(int fd, const void* data, size_t n) {
    const char* p = static_cast<const char*>(data);
    while (n > 0) {
        const ssize_t k = ::send(fd, p, n, MSG_NOSIGNAL);
        if (k <= 0) return false;
        p += k;
        n -= static_cast<size_t>(k);
    }
    return true;
}

bool send_str(int fd, const std::string& s) { return send_all(fd, s.data(), s.size()); }

const char* kPage =
    "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>G1 person tracking</title>"
    "<style>body{margin:0;background:#1a1a19;color:#c3c2b7;font:14px system-ui,-apple-system,sans-serif}"
    "main{max-width:1280px;margin:0 auto;padding:12px}img{width:100%;height:auto;display:block;"
    "background:#000;border-radius:4px}p{margin:8px 2px}</style></head><body><main>"
    "<img src=\"/stream\" alt=\"live view\">"
    "<p>Green: the person being followed. Grey: other tracked people. Status line: frame, results per second, "
    "latency from camera frame to result, tracked people, target ID.</p></main></body></html>";

}  // namespace

LiveView::LiveView(int port, double max_fps, int frame_w, int frame_h, PixelFormat fmt, float box_scale,
                   int jpeg_quality)
    : port_(port),
      min_interval_(max_fps > 0 ? 1.0 / max_fps : 0.0),
      w_(frame_w),
      h_(frame_h),
      fmt_(fmt),
      box_scale_(box_scale),
      jpeg_quality_(jpeg_quality) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        std::fprintf(stderr, "[view] no socket (%s); live view off\n", std::strerror(errno));
        return;
    }
    int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 || ::listen(fd, 8) != 0) {
        std::fprintf(stderr, "[view] cannot listen on port %d (%s); live view off (try --view-port)\n", port,
                     std::strerror(errno));
        ::close(fd);
        return;
    }
    listen_fd_ = fd;
    // print the addresses to open
    ifaddrs* ifs = nullptr;
    if (::getifaddrs(&ifs) == 0) {
        for (ifaddrs* i = ifs; i; i = i->ifa_next) {
            if (!i->ifa_addr || i->ifa_addr->sa_family != AF_INET) continue;
            char ip[INET_ADDRSTRLEN] = {0};
            ::inet_ntop(AF_INET, &reinterpret_cast<sockaddr_in*>(i->ifa_addr)->sin_addr, ip, sizeof(ip));
            if (std::strncmp(ip, "127.", 4) == 0) continue;
            std::fprintf(stderr, "[view] live view: http://%s:%d/   (%s)\n", ip, port, i->ifa_name);
        }
        ::freeifaddrs(ifs);
    }
    accept_thread_ = std::thread(&LiveView::accept_loop, this);
    encode_thread_ = std::thread(&LiveView::encode_loop, this);
}

LiveView::~LiveView() {
    running_ = false;
    in_cv_.notify_all();
    out_cv_.notify_all();
    {
        std::lock_guard<std::mutex> lk(conn_m_);
        for (int fd : clients_) ::shutdown(fd, SHUT_RDWR);  // unblocks any send()
    }
    if (accept_thread_.joinable()) accept_thread_.join();  // within 0.2 s (poll timeout)
    if (encode_thread_.joinable()) encode_thread_.join();
    if (listen_fd_ >= 0) ::close(listen_fd_);
    for (int i = 0; i < 300 && connections_.load() > 0; ++i)  // client threads use this object: let them end
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

bool LiveView::wants_frame(double now_s) const {
    return ok() && has_viewers() && now_s - last_submit_.load() >= min_interval_;
}

void LiveView::submit(const uint8_t* data, size_t bytes, const Overlay& ov, double now_s) {
    last_submit_ = now_s;
    {
        std::lock_guard<std::mutex> lk(in_m_);
        pending_.assign(data, data + bytes);
        pending_ov_ = ov;
        has_pending_ = true;
    }
    in_cv_.notify_one();
}

void LiveView::encode_loop() {
    const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, jpeg_quality_};
    std::vector<uint8_t> frame;
    Overlay ov;
    cv::Mat bgr;
    while (running_) {
        {
            std::unique_lock<std::mutex> lk(in_m_);
            in_cv_.wait_for(lk, std::chrono::milliseconds(500), [&] { return has_pending_ || !running_; });
            if (!running_) break;
            if (!has_pending_) continue;
            frame.swap(pending_);
            ov = pending_ov_;
            has_pending_ = false;
        }
        frame_to_bgr(frame.data(), w_, h_, fmt_, bgr);
        draw_overlay(bgr, ov, box_scale_);
        std::vector<unsigned char> jpg;
        if (!cv::imencode(".jpg", bgr, jpg, params)) continue;
        {
            std::lock_guard<std::mutex> lk(out_m_);
            jpeg_.swap(jpg);
            ++jpeg_id_;
        }
        ++encoded_;
        out_cv_.notify_all();
    }
}

void LiveView::accept_loop() {
    while (running_) {
        // poll with a timeout instead of blocking in accept(): closing a listening socket does not wake a
        // blocked accept() everywhere, and the destructor must be able to stop this thread
        pollfd pfd{listen_fd_, POLLIN, 0};
        if (::poll(&pfd, 1, 200) <= 0 || !(pfd.revents & POLLIN)) continue;
        const int fd = ::accept(listen_fd_, nullptr, nullptr);
        if (fd < 0) continue;
        if (!running_) {
            ::close(fd);
            break;
        }
        ++connections_;
        {
            std::lock_guard<std::mutex> lk(conn_m_);
            clients_.insert(fd);
        }
        std::thread(&LiveView::serve, this, fd).detach();
    }
}

void LiveView::serve(int fd) {
    timeval tv{5, 0};  // a client that never sends its request does not keep a thread forever
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    char req[2048];
    const ssize_t n = ::recv(fd, req, sizeof(req) - 1, 0);
    std::string path = "/";
    if (n > 0) {
        req[n] = 0;
        const char* sp = std::strchr(req, ' ');  // "GET /path HTTP/1.1"
        if (sp) {
            const char* end = std::strchr(sp + 1, ' ');
            if (end) path.assign(sp + 1, end);
        }
    }
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    auto latest = [&](uint64_t after, std::vector<unsigned char>& out, uint64_t& id) {
        std::unique_lock<std::mutex> lk(out_m_);
        out_cv_.wait_for(lk, std::chrono::seconds(1), [&] { return jpeg_id_ != after || !running_; });
        if (!running_ || jpeg_id_ == after) return false;
        out = jpeg_;
        id = jpeg_id_;
        return true;
    };

    if (n > 0 && path == "/stream") {
        ++viewers_;
        bool ok = send_str(fd, "HTTP/1.0 200 OK\r\nCache-Control: no-cache, no-store\r\nPragma: no-cache\r\n"
                               "Connection: close\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n");
        uint64_t sent = 0;
        std::vector<unsigned char> jpg;
        while (ok && running_) {
            if (!latest(sent, jpg, sent)) continue;
            char part[128];
            const int len = std::snprintf(part, sizeof(part),
                                          "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n",
                                          jpg.size());
            ok = send_all(fd, part, static_cast<size_t>(len)) && send_all(fd, jpg.data(), jpg.size()) &&
                 send_all(fd, "\r\n", 2);
        }
        --viewers_;
    } else if (n > 0 && path == "/snapshot.jpg") {
        ++viewers_;  // makes the tracker hand over a frame
        std::vector<unsigned char> jpg;
        uint64_t id = 0;
        bool got = false;
        for (int i = 0; i < 3 && !got && running_; ++i) got = latest(0, jpg, id);
        --viewers_;
        if (got) {
            send_str(fd, "HTTP/1.0 200 OK\r\nCache-Control: no-cache\r\nConnection: close\r\nContent-Type: image/jpeg\r\n"
                         "Content-Length: " + std::to_string(jpg.size()) + "\r\n\r\n");
            send_all(fd, jpg.data(), jpg.size());
        } else {
            send_str(fd, "HTTP/1.0 503 Service Unavailable\r\nConnection: close\r\n\r\nno frame yet\n");
        }
    } else if (n > 0 && (path == "/" || path == "/index.html")) {
        const std::string page(kPage);
        send_str(fd, "HTTP/1.0 200 OK\r\nConnection: close\r\nContent-Type: text/html; charset=utf-8\r\n"
                     "Content-Length: " + std::to_string(page.size()) + "\r\n\r\n" + page);
    } else if (n > 0) {
        send_str(fd, "HTTP/1.0 404 Not Found\r\nConnection: close\r\n\r\n");
    }
    {
        std::lock_guard<std::mutex> lk(conn_m_);
        clients_.erase(fd);
    }
    ::close(fd);
    --connections_;
}

}  // namespace pt
