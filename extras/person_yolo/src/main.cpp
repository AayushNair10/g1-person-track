// person_track: real-time single-person tracking on the Unitree G1 head camera (Jetson Orin NX,
// JetPack 5.1.1: CUDA 11.4, TensorRT 8.5, OpenCV 4.2). Nothing else is needed on the robot.
//
//   camera thread -> pinned frame -> GPU letterbox kernel -> TensorRT (YOLO26n) -> decode (+NMS for raw
//   heads) -> OC-SORT -> target selector -> CSV / stdout (+ optional annotated MP4)
//
// Run with --help for options. Per frame it prints:
//   frame,timestamp,target_id,x1,y1,x2,y2,conf,n_persons,wait_ms,upload_ms,pre_ms,infer_ms,post_ms,track_ms,total_ms,event
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "camera.hpp"
#include "live_view.hpp"
#include "cuda_utils.hpp"
#include "ocsort.hpp"
#include "postprocess.hpp"
#include "preprocess.hpp"
#include "target.hpp"
#include "timing.hpp"
#include "trt_engine.hpp"
#include "video_sink.hpp"

namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

struct Args {
    std::string engine = "models/yolo26n_640x384.fp16.engine";
    std::string source = "/dev/video4";  // Option B: the head camera's colour sensor directly
    std::string format = "auto";
    std::string dec_opts;
    std::string csv, summary, save_video, compare, label;
    double save_seconds = 30.0, duration = 0.0;
    long max_frames = 0;
    int warmup = 15;
    float conf = 0.25f, iou = 0.7f, relock_iou = 0.3f;
    bool graph = true, zero_copy = false, quiet = false, hw_resize = false;
    int print_every = 1, every = 1, dla = -1;
    int view_port = 8090;
    double view_fps = 15.0;
    int width = 0, height = 0, fps = 0;  // 0 = automatic (see resolve_camera_mode)
};

void usage(const char* prog) {
    std::printf(
        "usage: %s [options]\n"
        "  --engine PATH       TensorRT engine (default models/yolo26n_640x384.fp16.engine)\n"
        "  --source SRC        /dev/videoN | FILE.h264 | FILE.mp4 | videohub | videohub-nv12 | 'gst pipeline ! appsink'\n"
        "                      (default /dev/video4 = Option B: the head camera's colour sensor directly; needs\n"
        "                      videohub_pc4 stopped by hand first. videohub = Option A, Unitree's stream)\n"
        "  --format bgr|nv12   .h264 files: default nv12. /dev/videoN: default bgr (OpenCV V4L2); nv12 = GStreamer\n"
        "                      v4l2src with the VIC converting YUYV -> NV12 (no CPU colour conversion)\n"
        "  --hw-resize         videohub / .h264: the VIC (nvvidconv) scales frames to the model's picture size\n"
        "                      and hands over NV12 (e.g. 640x360 instead of 1280x720); boxes stay in camera pixels\n"
        "  --every N           process 1 of every N frames (others are skipped uncopied; default 1)\n"
        "  --dla N             run the engine on DLA core N (engines built with DLA=N scripts/build_engines.sh;\n"
        "                      default: taken from a '.dlaN.' in the engine file name)\n"
        "  --dec-opts STR      extra nvv4l2decoder properties, e.g. 'enable-max-performance=1 disable-dpb=1'\n"
        "  --conf F            person score threshold (default 0.25, kept if score > F)\n"
        "  --iou F             NMS IoU for raw-head models (default 0.7)\n"
        "  --relock-iou F      min IoU to re-lock the target after its ID is lost (default 0.3)\n"
        "  --csv PATH          per-frame CSV\n"
        "  --summary PATH      run summary (Markdown; PATH.json too)\n"
        "  --label TEXT        name of this run in the summary\n"
        "  --save-video PATH   annotated MP4 of the first --save-seconds (default 30) seconds\n"
        "  --view-port N       live view in a browser at http://<robot>:N/ (default 8090; 0 = off). Costs nothing\n"
        "                      while no browser is open\n"
        "  --view-fps F        frames per second sent to the browser (default 15)\n"
        "  --duration S        stop after S seconds (default: until the source ends / Ctrl+C)\n"
        "  --max-frames N      stop after N frames\n"
        "  --warmup N          frames left out of the timing statistics (default 15)\n"
        "  --compare PATH      also run this engine (e.g. the fp32 one) on every frame and report agreement\n"
        "  --no-graph          do not record the inference as a CUDA graph\n"
        "  --zero-copy         let the GPU read the pinned frame directly instead of copying it first\n"
        "  --print-every N     print every Nth frame to stdout (default 1)\n"
        "  --quiet             no per-frame lines on stdout\n"
        "  --width/--height    /dev/videoN: mode to ask for (default 640x360 = the 640x384 model's picture size);\n"
        "                      videohub: the stream's size (default 1280x720); files: read from the file\n"
        "  --fps N             /dev/videoN: frame rate to ask for (default 30)\n",
        prog);
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", k.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (k == "--help" || k == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else if (k == "--engine") a.engine = val();
        else if (k == "--source") a.source = val();
        else if (k == "--format") a.format = val();
        else if (k == "--hw-resize") a.hw_resize = true;
        else if (k == "--every") a.every = std::max(1, std::stoi(val()));
        else if (k == "--dla") a.dla = std::stoi(val());
        else if (k == "--dec-opts") a.dec_opts = val();
        else if (k == "--conf") a.conf = std::stof(val());
        else if (k == "--iou") a.iou = std::stof(val());
        else if (k == "--relock-iou") a.relock_iou = std::stof(val());
        else if (k == "--csv") a.csv = val();
        else if (k == "--summary") a.summary = val();
        else if (k == "--label") a.label = val();
        else if (k == "--save-video") a.save_video = val();
        else if (k == "--view-port") a.view_port = std::stoi(val());
        else if (k == "--view-fps") a.view_fps = std::stod(val());
        else if (k == "--save-seconds") a.save_seconds = std::stod(val());
        else if (k == "--duration") a.duration = std::stod(val());
        else if (k == "--max-frames") a.max_frames = std::stol(val());
        else if (k == "--warmup") a.warmup = std::stoi(val());
        else if (k == "--compare") a.compare = val();
        else if (k == "--no-graph") a.graph = false;
        else if (k == "--zero-copy") a.zero_copy = true;
        else if (k == "--print-every") a.print_every = std::max(1, std::stoi(val()));
        else if (k == "--quiet") a.quiet = true;
        else if (k == "--width") a.width = std::stoi(val());
        else if (k == "--height") a.height = std::stoi(val());
        else if (k == "--fps") a.fps = std::stoi(val());
        else {
            std::fprintf(stderr, "unknown option %s (see --help)\n", k.c_str());
            return false;
        }
    }
    return true;
}

// Fills in --width/--height/--fps left at 0: the camera mode to ask for (/dev/videoN), the stream's size
// (videohub), or for files the size read from the file (only needed for --hw-resize).
void resolve_camera_mode(Args& a) {
    const bool direct = a.source.rfind("/dev/video", 0) == 0;
    const bool hub = a.source.rfind("videohub", 0) == 0;
    if (a.fps <= 0) a.fps = direct ? 30 : 15;
    if (a.width > 0 && a.height > 0) return;
    if (direct) {
        a.width = 640;
        a.height = 360;
    } else if (!hub && a.hw_resize) {
        pt::CameraConfig c;
        c.source = a.source;
        c.format = a.format;
        c.dec_opts = a.dec_opts;
        if (!pt::probe_frame_size(c, a.width, a.height)) {
            std::fprintf(stderr, "cannot read the frame size of %s\n", a.source.c_str());
            std::exit(1);
        }
    } else {
        a.width = 1280;
        a.height = 720;
    }
}

double rss_mb() {  // resident memory of this process (Linux)
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("VmRSS:", 0) == 0) return std::atof(line.c_str() + 6) / 1024.0;
    return -1.0;
}

// DLA core from an engine file name like models/yolo26n_640x384.fp16.dla0.engine (-1 = GPU)
int dla_from_name(const std::string& path) {
    const size_t slash = path.rfind('/');
    const std::string base = slash == std::string::npos ? path : path.substr(slash + 1);
    const size_t p = base.find(".dla");
    if (p == std::string::npos || p + 4 >= base.size() || base[p + 4] < '0' || base[p + 4] > '9') return -1;
    return base[p + 4] - '0';
}

const char* head_name(pt::Head h) { return h == pt::Head::NmsFree ? "NMS-free (1x300x6)" : "raw + C++ NMS"; }

pt::OutputLayout layout_of(const pt::TrtEngine& e) {
    const auto& d = e.output().dims;
    pt::OutputLayout L{};
    if (!pt::layout_from_dims(d.d, d.nbDims, L)) {
        std::fprintf(stderr, "%s: unknown output shape (expected 1x300x6 or 1x(4+nc)xN)\n", e.path().c_str());
        std::exit(1);
    }
    return L;
}

// fp16 vs fp32 (or any two engines) on the same input tensor
struct Agreement {
    long frames = 0, same_count = 0, n_a = 0, n_b = 0, matched = 0;
    double iou_sum = 0, iou_min = 1, dconf_max = 0;
    std::vector<double> infer_b_ms;

    void add(const std::vector<pt::Det>& a, const std::vector<pt::Det>& b) {
        ++frames;
        same_count += a.size() == b.size();
        n_a += static_cast<long>(a.size());
        n_b += static_cast<long>(b.size());
        std::vector<char> used(b.size(), 0);  // greedy: best IoU partner for each box of a (a is score-sorted)
        for (const auto& x : a) {
            int best = -1;
            float bi = 0.5f;
            for (size_t j = 0; j < b.size(); ++j) {
                if (used[j]) continue;
                const float v = pt::box_iou(x, b[j]);
                if (v >= bi) {
                    bi = v;
                    best = static_cast<int>(j);
                }
            }
            if (best < 0) continue;
            used[best] = 1;
            ++matched;
            iou_sum += bi;
            iou_min = std::min(iou_min, static_cast<double>(bi));
            dconf_max = std::max(dconf_max, static_cast<double>(std::fabs(x.score - b[best].score)));
        }
    }
};

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) return 2;
    resolve_camera_mode(a);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    std::signal(SIGPIPE, SIG_IGN);  // a browser closing the live view must never stop the tracker

    pt::CameraConfig cc;
    cc.source = a.source;
    cc.format = a.format;
    cc.dec_opts = a.dec_opts;
    cc.width = a.width;
    cc.height = a.height;
    cc.fps = a.fps;
    cc.every = a.every;
    // Camera first: on the G1, master_service restarts a stopped videohub_pc4 within ~2-3 s, so the camera must
    // be taken right away (loading the engine takes longer). Only --hw-resize needs the engine's input size
    // first, to ask the VIC for the right picture size.
    std::unique_ptr<pt::Camera> cam_first;
    if (!a.hw_resize) cam_first.reset(new pt::Camera(cc));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const int dla = a.dla >= 0 ? a.dla : dla_from_name(a.engine);
    pt::TrtEngine eng(a.engine, stream, a.graph, dla);
    const pt::OutputLayout layout = layout_of(eng);

    if (a.hw_resize) {  // the picture inside the letterbox, at the camera's aspect ratio (even sizes for NV12)
        const pt::Letterbox at_cam = pt::make_letterbox(a.width, a.height, eng.input_w(), eng.input_h());
        cc.out_w = at_cam.new_w & ~1;
        cc.out_h = at_cam.new_h & ~1;
        cam_first.reset(new pt::Camera(cc));
    }
    pt::Camera& cam = *cam_first;
    // camera size: the frames' own size, or the stream's full size when the VIC scaled them
    const int cam_w = cam.hw_resized() ? a.width : cam.width(), cam_h = cam.hw_resized() ? a.height : cam.height();

    const pt::Letterbox lb = pt::make_letterbox(cam.width(), cam.height(), eng.input_w(), eng.input_h());
    const pt::Letterbox lb_box = pt::for_camera(lb, cam_w, cam_h);  // boxes back to camera pixels
    pt::Decoder dec(layout, lb_box, a.conf, a.iou);
    std::fprintf(stderr, "[app] engine %s: input %dx%d, output %s, %s, CUDA graph %s, frame copy %s\n",
                 a.engine.c_str(), eng.input_w(), eng.input_h(), head_name(layout.head),
                 dla >= 0 ? ("DLA core " + std::to_string(dla)).c_str() : "GPU", a.graph ? "on" : "off",
                 a.zero_copy ? "zero-copy" : "explicit");
    std::fprintf(stderr, "[app] letterbox %dx%d -> %dx%d at (%d, %d)%s\n", cam.width(), cam.height(), lb.new_w,
                 lb.new_h, lb.left, lb.top,
                 cam.hw_resized() ? (", boxes in camera pixels " + std::to_string(cam_w) + "x" + std::to_string(cam_h)).c_str()
                                  : "");

    std::unique_ptr<pt::TrtEngine> eng2;
    std::unique_ptr<pt::Decoder> dec2;
    if (!a.compare.empty()) {
        eng2.reset(new pt::TrtEngine(a.compare, stream, a.graph, dla_from_name(a.compare)));
        if (eng2->input_w() != eng.input_w() || eng2->input_h() != eng.input_h()) {
            std::fprintf(stderr, "--compare engine must have the same input size\n");
            return 2;
        }
        dec2.reset(new pt::Decoder(layout_of(*eng2), lb_box, a.conf, a.iou));
        std::fprintf(stderr, "[app] comparing with %s on every frame\n", a.compare.c_str());
    }

    // GPU copy of the frame (unless the kernel reads the pinned frame directly)
    uint8_t* frame_dev = nullptr;
    size_t frame_bytes = 0;
    cudaEvent_t ev[5];
    for (auto& e : ev) CUDA_CHECK(cudaEventCreate(&e));

    // Lost-track and target memory: 30 frames, as in the benchmark (Ultralytics counts frames; at the benchmark's
    // 30 FPS that is 1 s). Kept at >= 1 s for a direct camera above 30 FPS, and in seconds for --every N.
    const bool direct = a.source.rfind("/dev/video", 0) == 0;
    const double rate = direct ? (cam.source_fps() > 0 ? cam.source_fps() : a.fps) : 30.0;
    const double mem_scale = std::max(1.0, rate / 30.0) / a.every;
    pt::OCSortConfig oc;
    oc.track_buffer = std::max(1, static_cast<int>(std::lround(oc.track_buffer * mem_scale)));
    pt::OCSort tracker(oc);
    pt::TargetConfig tc;
    tc.relock_min_iou = a.relock_iou;
    tc.reacquire_after = std::max(1, static_cast<int>(std::lround(tc.reacquire_after * mem_scale)));
    if (oc.track_buffer != 30)
        std::fprintf(stderr, "[app] tracker memory %d frames, target re-acquire after %d frames\n", oc.track_buffer,
                     tc.reacquire_after);
    pt::TargetSelector target(tc);

    std::FILE* csv = nullptr;
    const char* header =
        "frame,timestamp,target_id,x1,y1,x2,y2,conf,n_persons,wait_ms,upload_ms,pre_ms,infer_ms,post_ms,track_ms,"
        "total_ms,event\n";
    if (!a.csv.empty()) {
        csv = std::fopen(a.csv.c_str(), "w");
        if (!csv) {
            std::fprintf(stderr, "cannot write %s\n", a.csv.c_str());
            return 1;
        }
        std::fputs(header, csv);
    }
    if (!a.quiet) std::fputs(header, stdout);

    std::unique_ptr<pt::VideoSink> sink;
    if (!a.save_video.empty())
        sink.reset(new pt::VideoSink(a.save_video, cam.width(), cam.height(), cam.format(),
                                     (cam.source_fps() > 0 ? cam.source_fps() : 15.0) / a.every, a.save_seconds,
                                     static_cast<float>(cam.width()) / static_cast<float>(cam_w)));

    std::unique_ptr<pt::LiveView> view;
    if (a.view_port > 0) {
        view.reset(new pt::LiveView(a.view_port, a.view_fps, cam.width(), cam.height(), cam.format(),
                                    static_cast<float>(cam.width()) / static_cast<float>(cam_w)));
        if (!view->ok()) view.reset();
    }

    std::vector<double> s_wait, s_upload, s_pre, s_infer, s_post, s_track, s_total, s_gpu_wall;
    std::vector<pt::Det> dets, dets2;
    Agreement agree;
    std::set<int> ids_seen;
    long frames = 0, with_target = 0, without_target = 0, counted = 0;
    double t_first = 0, t_last = 0, fps_ema = 0, t_prev = 0, t_status = 0;
    uint64_t last_seq = 0, gaps = 0, read_base = 0, drop_base = 0;
    const double t_start = pt::now_s();
    const double t_start_unix =  // same moment on the system clock, to line up with tegrastats / other logs
        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();

    while (!g_stop) {
        const double t_req = pt::now_s();
        pt::Frame fr;
        int got = 0;
        double t_warn = t_req;
        while (!g_stop && (got = cam.next(fr, 100)) == 0) {  // short waits so Ctrl+C always works
            if (pt::now_s() - t_warn > 2.0) {
                std::fprintf(stderr, "[app] no frame from the source for 2 s...\n");
                t_warn = pt::now_s();
            }
        }
        if (got != 1) break;
        const double t_got = pt::now_s();
        if (last_seq && fr.seq > last_seq + 1 && frames >= a.warmup) gaps += fr.seq - last_seq - 1;
        last_seq = fr.seq;

        // ---- GPU: upload, letterbox, inference, download (one stream, one wait)
        const uint8_t* src = nullptr;
        CUDA_CHECK(cudaEventRecord(ev[0], stream));
        if (a.zero_copy) {
            void* p = nullptr;
            CUDA_CHECK(cudaHostGetDevicePointer(&p, const_cast<uint8_t*>(fr.data), 0));
            src = static_cast<const uint8_t*>(p);
        } else {
            if (fr.bytes != frame_bytes) {
                if (frame_dev) CUDA_CHECK(cudaFree(frame_dev));
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&frame_dev), fr.bytes));
                frame_bytes = fr.bytes;
            }
            CUDA_CHECK(cudaMemcpyAsync(frame_dev, fr.data, fr.bytes, cudaMemcpyHostToDevice, stream));
            src = frame_dev;
        }
        CUDA_CHECK(cudaEventRecord(ev[1], stream));
        pt::letterbox_gpu(src, fr.stride, fr.fmt, lb, eng.input().dev, stream);
        CUDA_CHECK(cudaEventRecord(ev[2], stream));
        eng.enqueue();
        CUDA_CHECK(cudaEventRecord(ev[3], stream));
        eng.download();
        if (eng2) {
            CUDA_CHECK(cudaMemcpyAsync(eng2->input().dev, eng.input().dev, eng.input().count * sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream));
            eng2->enqueue();
            CUDA_CHECK(cudaEventRecord(ev[4], stream));
            eng2->download();
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const double t_gpu = pt::now_s();
        float ms_upload = 0, ms_pre = 0, ms_infer = 0, ms_infer2 = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms_upload, ev[0], ev[1]));
        CUDA_CHECK(cudaEventElapsedTime(&ms_pre, ev[1], ev[2]));
        CUDA_CHECK(cudaEventElapsedTime(&ms_infer, ev[2], ev[3]));
        if (eng2) CUDA_CHECK(cudaEventElapsedTime(&ms_infer2, ev[3], ev[4]));

        // ---- CPU: decode, track, choose target
        dec.decode(eng.output().host, dets);
        if (eng2) {
            dec2->decode(eng2->output().host, dets2);
            agree.add(dets, dets2);
        }
        const double t_post = pt::now_s();
        const auto& tracks = tracker.update(dets);
        pt::TrackOut tgt{};
        const bool has = target.update(tracks, tgt);
        const double t_done = pt::now_s();
        for (const auto& t : tracks) ids_seen.insert(t.id);

        ++frames;
        has ? ++with_target : ++without_target;
        const double total_ms = 1000.0 * (t_done - fr.t_ready);
        if (t_prev > 0) {
            const double inst = 1.0 / std::max(1e-6, t_done - t_prev);
            fps_ema = fps_ema > 0 ? 0.9 * fps_ema + 0.1 * inst : inst;
        }
        t_prev = t_done;

        const bool to_sink = sink && sink->wants_more(), to_view = view && view->wants_frame(t_done);
        if (to_sink || to_view) {
            pt::Overlay ov;
            ov.frame = static_cast<uint64_t>(frames);
            ov.tracks = tracks;
            ov.has_target = has;
            ov.target = tgt;
            ov.fps = fps_ema;
            ov.latency_ms = total_ms;
            if (to_sink) sink->push(fr.data, fr.bytes, ov);
            if (to_view) view->submit(fr.data, fr.bytes, ov, t_done);
        }
        cam.release(fr);

        if (frames > a.warmup) {
            if (counted == 0) {  // statistics start here (engine loading made the first frames late)
                t_first = t_got;
                read_base = cam.frames_read();
                drop_base = cam.frames_dropped();
            }
            t_last = t_done;
            ++counted;
            s_wait.push_back(1000.0 * (t_got - t_req));
            s_upload.push_back(ms_upload);
            s_pre.push_back(ms_pre);
            s_infer.push_back(ms_infer);
            s_gpu_wall.push_back(1000.0 * (t_gpu - t_got));
            s_post.push_back(1000.0 * (t_post - t_gpu));
            s_track.push_back(1000.0 * (t_done - t_post));
            s_total.push_back(total_ms);
            if (eng2) agree.infer_b_ms.push_back(ms_infer2);
        }

        const char* evs = "-";
        switch (target.last_event()) {
            case pt::TargetSelector::Event::Acquired: evs = "acquired"; break;
            case pt::TargetSelector::Event::Kept: evs = "kept"; break;
            case pt::TargetSelector::Event::Relocked: evs = "relocked"; break;
            case pt::TargetSelector::Event::Lost: evs = "lost"; break;
            default: break;
        }
        char row[512];
        const double ts = fr.t_ready - t_start;
        if (has)
            std::snprintf(row, sizeof(row), "%ld,%.4f,%d,%.1f,%.1f,%.1f,%.1f,%.3f,%zu,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%s\n",
                          frames, ts, tgt.id, tgt.x1, tgt.y1, tgt.x2, tgt.y2, tgt.score, dets.size(),
                          1000.0 * (t_got - t_req), ms_upload, ms_pre, ms_infer, 1000.0 * (t_post - t_gpu),
                          1000.0 * (t_done - t_post), total_ms, evs);
        else
            std::snprintf(row, sizeof(row), "%ld,%.4f,-1,,,,,,%zu,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%s\n", frames, ts,
                          dets.size(), 1000.0 * (t_got - t_req), ms_upload, ms_pre, ms_infer,
                          1000.0 * (t_post - t_gpu), 1000.0 * (t_done - t_post), total_ms, evs);
        if (csv) std::fputs(row, csv);
        if (!a.quiet && frames % a.print_every == 0) std::fputs(row, stdout);
        if (a.quiet && t_done - t_status >= 2.0) {
            t_status = t_done;
            std::fprintf(stderr, "[app] frame %ld  %.1f FPS  latency %.1f ms  infer %.2f ms  target %d  persons %zu\n",
                         frames, fps_ema, total_ms, ms_infer, has ? tgt.id : -1, dets.size());
        }

        if (a.max_frames > 0 && frames >= a.max_frames) break;
        if (a.duration > 0 && t_done - t_start >= a.duration) break;
    }
    cam.stop();
    sink.reset();  // finish the video
    const unsigned long long view_frames = view ? view->frames_sent() : 0ULL;
    view.reset();
    if (csv) std::fclose(csv);

    // ---- summary
    const double span = t_last - t_first;
    const double fps = counted > 1 && span > 0 ? (counted - 1) / span : 0.0;
    const pt::Stat st_wait = pt::summarize(s_wait), st_up = pt::summarize(s_upload), st_pre = pt::summarize(s_pre),
                   st_inf = pt::summarize(s_infer), st_gw = pt::summarize(s_gpu_wall), st_post = pt::summarize(s_post),
                   st_trk = pt::summarize(s_track), st_tot = pt::summarize(s_total);
    const double rss = rss_mb();

    std::string md;
    char buf[1024];
    auto add = [&](const char* fmt, auto... xs) {
        std::snprintf(buf, sizeof(buf), fmt, xs...);
        md += buf;
    };
    add("## person_track run%s%s\n\n", a.label.empty() ? "" : ": ", a.label.c_str());
    const std::string where = dla >= 0 ? "DLA core " + std::to_string(dla) : "GPU";
    add("- engine `%s` (input %dx%d, %s, %s, CUDA graph %s, frame copy %s)\n", a.engine.c_str(), eng.input_w(),
        eng.input_h(), head_name(layout.head), where.c_str(), eng.graph_active() ? "on" : "off",
        a.zero_copy ? "zero-copy" : "explicit");
    add("- source %s, frames %dx%d %s%s, 1 of every %d processed\n", cam.description().c_str(), cam.width(),
        cam.height(), cam.format() == pt::PixelFormat::NV12 ? "NV12" : "BGR",
        cam.hw_resized() ? " (hardware resize; boxes in camera pixels)" : "", a.every);
    add("- frames processed %ld (statistics over %ld after %d warm-up), %.1f s\n", frames, counted, a.warmup,
        pt::now_s() - t_start);
    const unsigned long long n_read = cam.frames_read() - read_base, n_drop = cam.frames_dropped() - drop_base;
    add("- **throughput %.2f FPS**; after warm-up: frames read from source %llu, dropped as stale %llu "
        "(camera rate %.2f FPS)\n",
        fps, n_read, n_drop, cam.capture_fps());
    add("- target present in %ld frames, absent in %ld; target ID changes %d (re-locks %d, re-acquisitions %d); "
        "tracker IDs seen %zu\n",
        with_target, without_target, target.id_changes(), target.relocks(), target.reacquires(), ids_seen.size());
    add("- process RSS %.0f MB; live view frames sent %llu\n\n", rss, view_frames);
    md += "| stage (ms) | mean | p50 | p95 | max |\n|---|---|---|---|---|\n";
    auto row = [&](const char* name, const pt::Stat& s) {
        add("| %s | %.3f | %.3f | %.3f | %.3f |\n", name, s.mean, s.p50, s.p95, s.max);
    };
    row("wait for frame", st_wait);
    row("upload (GPU)", st_up);
    row("preprocess (GPU)", st_pre);
    row("inference (GPU)", st_inf);
    row("GPU section wall", st_gw);
    row("postprocess (CPU)", st_post);
    row("tracker + target (CPU)", st_trk);
    row("**latency: frame read -> result**", st_tot);
    if (eng2 && agree.frames > 0) {
        const pt::Stat sb = pt::summarize(agree.infer_b_ms);
        add("\n**Agreement with `%s`** over %ld frames: detections %ld vs %ld, matched (IoU >= 0.5) %ld, "
            "mean IoU %.4f, min IoU %.4f, max |score diff| %.4f, same count in %.1f%% of frames; "
            "its inference p50 %.3f ms\n",
            a.compare.c_str(), agree.frames, agree.n_a, agree.n_b, agree.matched,
            agree.matched ? agree.iou_sum / static_cast<double>(agree.matched) : 0.0, agree.iou_min, agree.dconf_max,
            100.0 * static_cast<double>(agree.same_count) / static_cast<double>(agree.frames), sb.p50);
    }
    std::fprintf(stderr, "\n%s\n", md.c_str());

    if (!a.summary.empty()) {
        if (std::FILE* f = std::fopen(a.summary.c_str(), "w")) {
            std::fputs(md.c_str(), f);
            std::fclose(f);
        }
        if (std::FILE* f = std::fopen((a.summary + ".json").c_str(), "w")) {
            auto js = [&](const char* k, const pt::Stat& s) {
                std::fprintf(f, "\"%s\":{\"mean\":%.4f,\"p50\":%.4f,\"p95\":%.4f,\"max\":%.4f},", k, s.mean, s.p50,
                             s.p95, s.max);
            };
            std::fprintf(f, "{\"label\":\"%s\",\"engine\":\"%s\",\"source\":\"%s\",\"input\":\"%dx%d\",\"head\":\"%s\",",
                         a.label.c_str(), a.engine.c_str(), a.source.c_str(), eng.input_w(), eng.input_h(),
                         head_name(layout.head));
            std::fprintf(f, "\"cuda_graph\":%s,\"zero_copy\":%s,\"frames\":%ld,\"counted\":%ld,\"fps\":%.3f,",
                         eng.graph_active() ? "true" : "false", a.zero_copy ? "true" : "false", frames, counted, fps);
            std::fprintf(f, "\"frame_size\":\"%dx%d\",\"hw_resize\":%s,\"every\":%d,\"dla\":%d,", cam.width(),
                         cam.height(), cam.hw_resized() ? "true" : "false", a.every, dla);
            std::fprintf(f, "\"view_frames\":%llu,", view_frames);
            std::fprintf(f, "\"camera_mode\":\"%dx%d@%d\",\"source_fps\":%.2f,\"camera_fps\":%.2f,\"start_unix\":%.3f,",
                         a.width, a.height, a.fps, cam.source_fps(), cam.capture_fps(), t_start_unix);
            std::fprintf(f, "\"frames_read\":%llu,\"dropped\":%llu,\"gaps\":%llu,\"with_target\":%ld,\"without_target\":%ld,",
                         n_read, n_drop, static_cast<unsigned long long>(gaps),
                         with_target, without_target);
            std::fprintf(f, "\"id_changes\":%d,\"relocks\":%d,\"reacquires\":%d,\"ids_seen\":%zu,\"rss_mb\":%.1f,",
                         target.id_changes(), target.relocks(), target.reacquires(), ids_seen.size(), rss);
            js("wait", st_wait);
            js("upload", st_up);
            js("pre", st_pre);
            js("infer", st_inf);
            js("gpu_wall", st_gw);
            js("post", st_post);
            js("track", st_trk);
            js("total", st_tot);
            if (eng2 && agree.frames > 0)
                std::fprintf(f, "\"compare\":{\"engine\":\"%s\",\"frames\":%ld,\"n_a\":%ld,\"n_b\":%ld,\"matched\":%ld,"
                             "\"iou_mean\":%.5f,\"iou_min\":%.5f,\"dconf_max\":%.5f,\"same_count_pct\":%.2f},",
                             a.compare.c_str(), agree.frames, agree.n_a, agree.n_b, agree.matched,
                             agree.matched ? agree.iou_sum / static_cast<double>(agree.matched) : 0.0, agree.iou_min,
                             agree.dconf_max,
                             100.0 * static_cast<double>(agree.same_count) / static_cast<double>(agree.frames));
            std::fprintf(f, "\"ok\":true}\n");
            std::fclose(f);
        }
    }
    for (auto& e : ev) cudaEventDestroy(e);
    if (frame_dev) cudaFree(frame_dev);
    eng2.reset();
    cudaStreamDestroy(stream);
    return 0;
}
