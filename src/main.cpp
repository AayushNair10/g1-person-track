// face_gesture: face detection and the hand gestures of the followed person, on the Unitree G1 head camera
// (Jetson Orin NX, JetPack 5.1.1: CUDA 11.4, TensorRT 8.5, OpenCV 4.2). Nothing else is needed on the robot.
//
//   camera thread -> pinned frame -> GPU (uploaded once)
//   faces: letterbox -> YuNet (TensorRT, decode + best 100 inside) -> NMS -> OC-SORT -> followed face
//   hands: only around the followed face. Hands already followed: rotated crop from last frame's landmarks ->
//          21 landmarks (TensorRT). Fewer than 2: palm detector on a region around the face -> new hands.
//          Landmarks -> 15 gestures (rules) -> smoothed -> events.
//   -> CSV / stdout, gesture events, live view in a browser, annotated MP4, run summary
//
// Run with --help for options.
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
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "camera.hpp"
#include "cuda_utils.hpp"
#include "face_detector.hpp"
#include "gestures.hpp"
#include "hands.hpp"
#include "live_view.hpp"
#include "ocsort.hpp"
#include "preprocess.hpp"
#include "target.hpp"
#include "timing.hpp"
#include "trt_engine.hpp"
#include "video_sink.hpp"

namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

struct Args {
    std::string face_engine = "models/yunet_640x384.fp16.engine";
    std::string palm_engine = "models/palm_192.fp16.engine";
    std::string hand_engine = "models/handpose_224.fp16.engine";
    std::string source = "/dev/video4";
    std::string format = "auto";
    std::string dec_opts;
    std::string csv, summary, save_video, label;
    double save_seconds = 30.0, duration = 0.0;
    long max_frames = 0;
    int warmup = 15;
    float face_conf = 0.6f, palm_conf = 0.5f, hand_keep = 0.5f, hand_new = 0.7f, relock_iou = 0.3f;
    float region_width = 6.f, region_above = 1.f, region_below = 3.5f;
    int search_every = 5;
    bool hands = true, mirrored = false;
    bool graph = true, zero_copy = false, quiet = false;
    int print_every = 1, every = 1;
    int view_port = 8090;
    double view_fps = 15.0;
    int width = 0, height = 0, fps = 0;  // 0 = automatic
};

void usage(const char* prog) {
    std::printf(
        "usage: %s [options]\n"
        "  --source SRC        /dev/videoN | FILE.h264 | FILE.mp4 | videohub | 'gst pipeline ! appsink'\n"
        "                      (default /dev/video4: the head camera's colour sensor directly; needs videohub_pc4\n"
        "                      stopped on the same command line: sudo pkill -f videohub_pc4; build/face_gesture)\n"
        "  --width/--height/--fps  camera mode (default 1280x720 @ 30: hands need the pixels)\n"
        "  --format bgr|nv12   /dev/videoN: bgr (OpenCV V4L2, default) or nv12 (GStreamer, VIC colour conversion)\n"
        "  --dec-opts STR      extra nvv4l2decoder properties (.h264 files, videohub)\n"
        "  --face-engine PATH  YuNet engine (default models/yunet_640x384.fp16.engine; also 416x256, 320x192)\n"
        "  --palm-engine PATH  palm detector engine (default models/palm_192.fp16.engine)\n"
        "  --hand-engine PATH  hand landmark engine (default models/handpose_224.fp16.engine)\n"
        "  --no-hands          faces only\n"
        "  --face-conf F       face score threshold (default 0.6)\n"
        "  --palm-conf F       palm score threshold (default 0.5)\n"
        "  --hand-keep F       keep following a hand while its presence is >= F (default 0.5)\n"
        "  --hand-new F        accept a new hand when its presence is >= F (default 0.7)\n"
        "  --search-every N    with one hand followed, look for the second every N frames (default 5)\n"
        "  --region W,A,B      hand search region around the followed face, in face sizes: width W, A above,\n"
        "                      B below (default 6,1,3.5)\n"
        "  --mirrored          the camera image is mirrored (selfie style); default: not (the robot's camera)\n"
        "  --relock-iou F      min IoU to re-lock the followed face after its ID is lost (default 0.3)\n"
        "  --csv PATH          per-frame CSV\n"
        "  --summary PATH      run summary (Markdown; PATH.json too)\n"
        "  --label TEXT        name of this run in the summary\n"
        "  --save-video PATH   annotated MP4 of the first --save-seconds (default 30) seconds\n"
        "  --view-port N       live view in a browser at http://<robot>:N/ (default 8090; 0 = off)\n"
        "  --view-fps F        frames per second sent to the browser (default 15)\n"
        "  --duration S        stop after S seconds\n"
        "  --max-frames N      stop after N frames\n"
        "  --every N           process 1 of every N frames (default 1)\n"
        "  --warmup N          frames left out of the timing statistics (default 15)\n"
        "  --no-graph          do not record the inferences as CUDA graphs\n"
        "  --zero-copy         let the GPU read the pinned frame directly instead of copying it first\n"
        "  --print-every N     print every Nth frame to stdout (default 1)\n"
        "  --quiet             no per-frame lines on stdout (gesture events are still printed)\n",
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
        } else if (k == "--source") a.source = val();
        else if (k == "--width") a.width = std::stoi(val());
        else if (k == "--height") a.height = std::stoi(val());
        else if (k == "--fps") a.fps = std::stoi(val());
        else if (k == "--format") a.format = val();
        else if (k == "--dec-opts") a.dec_opts = val();
        else if (k == "--face-engine") a.face_engine = val();
        else if (k == "--palm-engine") a.palm_engine = val();
        else if (k == "--hand-engine") a.hand_engine = val();
        else if (k == "--no-hands") a.hands = false;
        else if (k == "--face-conf") a.face_conf = std::stof(val());
        else if (k == "--palm-conf") a.palm_conf = std::stof(val());
        else if (k == "--hand-keep") a.hand_keep = std::stof(val());
        else if (k == "--hand-new") a.hand_new = std::stof(val());
        else if (k == "--search-every") a.search_every = std::max(1, std::stoi(val()));
        else if (k == "--region") {
            if (std::sscanf(val().c_str(), "%f,%f,%f", &a.region_width, &a.region_above, &a.region_below) != 3) {
                std::fprintf(stderr, "--region needs W,A,B\n");
                return false;
            }
        } else if (k == "--mirrored") a.mirrored = true;
        else if (k == "--relock-iou") a.relock_iou = std::stof(val());
        else if (k == "--csv") a.csv = val();
        else if (k == "--summary") a.summary = val();
        else if (k == "--label") a.label = val();
        else if (k == "--save-video") a.save_video = val();
        else if (k == "--save-seconds") a.save_seconds = std::stod(val());
        else if (k == "--view-port") a.view_port = std::stoi(val());
        else if (k == "--view-fps") a.view_fps = std::stod(val());
        else if (k == "--duration") a.duration = std::stod(val());
        else if (k == "--max-frames") a.max_frames = std::stol(val());
        else if (k == "--every") a.every = std::max(1, std::stoi(val()));
        else if (k == "--warmup") a.warmup = std::stoi(val());
        else if (k == "--no-graph") a.graph = false;
        else if (k == "--zero-copy") a.zero_copy = true;
        else if (k == "--print-every") a.print_every = std::max(1, std::stoi(val()));
        else if (k == "--quiet") a.quiet = true;
        else {
            std::fprintf(stderr, "unknown option %s (see --help)\n", k.c_str());
            return false;
        }
    }
    const bool direct = a.source.rfind("/dev/video", 0) == 0;
    if (a.fps <= 0) a.fps = direct ? 30 : 15;
    if (a.width <= 0 || a.height <= 0) {
        a.width = 1280;
        a.height = 720;
    }
    return true;
}

double rss_mb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("VmRSS:", 0) == 0) return std::atof(line.c_str() + 6) / 1024.0;
    return -1.0;
}

void need_output(const pt::TrtEngine& e, size_t count, const char* what) {
    if (e.output().count != count) {
        std::fprintf(stderr, "%s: output has %zu values, expected %zu (%s; export with scripts/export_face_hand.py)\n",
                     e.path().c_str(), e.output().count, count, what);
        std::exit(1);
    }
}

struct HandTrack {
    int id = 0;
    pt::RotRect roi{};
    pt::HandLandmarks lm{};
    float side = 0.5f;  // smoothed model handedness
    pt::GestureSmoother smooth;
    std::string reported;
    int age = 0;
};

// The person's own side ("left"/"right"). The hand model assumes a mirrored (selfie) image.
const char* person_side(float model_handedness, bool mirrored) {
    const bool model_right = model_handedness > 0.5f;
    return (mirrored ? model_right : !model_right) ? "right" : "left";
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) return 2;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    std::signal(SIGPIPE, SIG_IGN);  // a browser closing the live view must never stop the tracker

    // Camera first: master_service restarts a stopped videohub_pc4 within ~2-3 s, so take the camera now.
    pt::CameraConfig cc;
    cc.source = a.source;
    cc.format = a.format;
    cc.dec_opts = a.dec_opts;
    cc.width = a.width;
    cc.height = a.height;
    cc.fps = a.fps;
    cc.every = a.every;
    pt::Camera cam(cc);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    pt::TrtEngine face(a.face_engine, stream, a.graph);
    need_output(face, pt::FaceDecoder::kRows * pt::FaceDecoder::kCols, "YuNet 1x100x15");
    std::unique_ptr<pt::TrtEngine> palm, hand;
    if (a.hands) {
        palm.reset(new pt::TrtEngine(a.palm_engine, stream, a.graph));
        hand.reset(new pt::TrtEngine(a.hand_engine, stream, a.graph));
        need_output(*palm, pt::PalmDecoder::kRows * pt::PalmDecoder::kCols, "palm 1x32x21");
        need_output(*hand, 128, "hand 1x128");
    }
    const int palm_size = palm ? palm->input_w() : 192, hand_size = hand ? hand->input_w() : 224;

    pt::InputFormat face_in;  // YuNet: BGR, 0..255, planar
    face_in.nhwc = false;
    face_in.rgb = false;
    face_in.scale = 1.f;
    pt::InputFormat mp_in;  // MediaPipe: RGB, 0..1, interleaved, black outside
    mp_in.nhwc = true;
    mp_in.rgb = true;
    mp_in.scale = 1.f / 255.f;

    const int W = cam.width(), H = cam.height();
    const pt::Letterbox lb_face = pt::make_letterbox(W, H, face.input_w(), face.input_h());
    pt::FaceDecoder face_dec(lb_face, W, H, a.face_conf);
    pt::PalmDecoder palm_dec(a.palm_conf);
    const std::string hands_desc = a.hands ? a.palm_engine + " + " + a.hand_engine : std::string("off");
    std::fprintf(stderr, "[app] faces: %s (input %dx%d, picture %dx%d); hands: %s\n", a.face_engine.c_str(),
                 face.input_w(), face.input_h(), lb_face.new_w, lb_face.new_h, hands_desc.c_str());

    float* hand_out[4] = {nullptr, nullptr, nullptr, nullptr};  // followed 0-1, new 2-3
    for (auto& p : hand_out) CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&p), 128 * sizeof(float)));

    uint8_t* frame_dev = nullptr;
    size_t frame_bytes = 0;
    cudaEvent_t ev[5];
    for (auto& e : ev) CUDA_CHECK(cudaEventCreate(&e));

    // Lost-face memory: 30 frames as in the benchmark (1 s at 30 FPS), at least 1 s for faster cameras.
    const bool direct = a.source.rfind("/dev/video", 0) == 0;
    const double rate = direct ? (cam.source_fps() > 0 ? cam.source_fps() : a.fps) : 30.0;
    const double mem_scale = std::max(1.0, rate / 30.0) / a.every;
    pt::OCSortConfig oc;
    oc.track_buffer = std::max(1, static_cast<int>(std::lround(oc.track_buffer * mem_scale)));
    pt::OCSort tracker(oc);
    pt::TargetConfig tc;
    tc.relock_min_iou = a.relock_iou;
    tc.reacquire_after = std::max(1, static_cast<int>(std::lround(tc.reacquire_after * mem_scale)));
    pt::TargetSelector target(tc);

    std::FILE* csv = nullptr;
    const char* header =
        "frame,timestamp,target_id,x1,y1,x2,y2,conf,n_faces,n_hands,left_hand,right_hand,wait_ms,upload_ms,pre_ms,"
        "infer_ms,hand_ms,post_ms,track_ms,total_ms,event\n";
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
        sink.reset(new pt::VideoSink(a.save_video, W, H, cam.format(),
                                     (cam.source_fps() > 0 ? cam.source_fps() : a.fps) / a.every, a.save_seconds));
    std::unique_ptr<pt::LiveView> view;
    if (a.view_port > 0) {
        view.reset(new pt::LiveView(a.view_port, a.view_fps, W, H, cam.format(), 1.f));
        if (!view->ok()) view.reset();
    }

    std::vector<double> s_wait, s_upload, s_pre, s_infer, s_hand, s_post, s_track, s_total, s_gpu_wall;
    std::vector<pt::Face> faces;
    std::vector<pt::Det> dets;
    std::vector<pt::Palm> palms;
    std::vector<HandTrack> hands;
    std::map<std::string, int> gesture_counts;
    std::set<int> ids_seen;
    long frames = 0, with_target = 0, without_target = 0, counted = 0, with_hand = 0, palm_searches = 0;
    int next_hand_id = 1;
    double t_first = 0, t_last = 0, fps_ema = 0, t_prev = 0, t_status = 0;
    uint64_t last_seq = 0, gaps = 0, read_base = 0, drop_base = 0;
    const double t_start = pt::now_s();
    const double t_start_unix =
        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();

    while (!g_stop) {
        const double t_req = pt::now_s();
        pt::Frame fr;
        int got = 0;
        double t_warn = t_req;
        while (!g_stop && (got = cam.next(fr, 100)) == 0) {
            if (pt::now_s() - t_warn > 2.0) {
                std::fprintf(stderr, "[app] no frame from the source for 2 s...\n");
                t_warn = pt::now_s();
            }
        }
        if (got != 1) break;
        const double t_got = pt::now_s();
        if (last_seq && fr.seq > last_seq + 1 && frames >= a.warmup) gaps += fr.seq - last_seq - 1;
        last_seq = fr.seq;

        // ---- GPU batch: upload, faces, hands already followed (crops from last frame's landmarks)
        CUDA_CHECK(cudaEventRecord(ev[0], stream));
        const uint8_t* src = nullptr;
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
        pt::GpuFrame gf;
        gf.data = src;
        gf.stride = fr.stride;
        gf.w = fr.w;
        gf.h = fr.h;
        gf.fmt = fr.fmt;
        CUDA_CHECK(cudaEventRecord(ev[1], stream));
        pt::letterbox_gpu(gf, lb_face, face_in, face.input().dev, stream);
        CUDA_CHECK(cudaEventRecord(ev[2], stream));
        face.enqueue();
        face.download();
        CUDA_CHECK(cudaEventRecord(ev[3], stream));
        const size_t n_followed = hands.size();
        for (size_t i = 0; i < n_followed; ++i) {
            pt::rotcrop_gpu(gf, hands[i].roi, hand_size, mp_in, hand->input().dev, stream);
            hand->enqueue();
            hand->download(hand_out[i]);
        }
        CUDA_CHECK(cudaEventRecord(ev[4], stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const double t_gpu = pt::now_s();
        float ms_upload = 0, ms_pre = 0, ms_infer = 0, ms_hand = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms_upload, ev[0], ev[1]));
        CUDA_CHECK(cudaEventElapsedTime(&ms_pre, ev[1], ev[2]));
        CUDA_CHECK(cudaEventElapsedTime(&ms_infer, ev[2], ev[3]));
        CUDA_CHECK(cudaEventElapsedTime(&ms_hand, ev[3], ev[4]));

        // ---- faces: decode, track, followed face
        face_dec.decode(face.output().host, faces);
        const double t_post = pt::now_s();
        dets.clear();
        for (const auto& f : faces) dets.push_back(pt::Det{f.x1, f.y1, f.x2, f.y2, f.score, 0});
        const auto& tracks = tracker.update(dets);
        pt::TrackOut tgt{};
        const bool has = target.update(tracks, tgt);
        for (const auto& t : tracks) ids_seen.insert(t.id);
        const pt::Face* tface = has && tgt.det_index >= 0 && tgt.det_index < static_cast<int>(faces.size())
                                    ? &faces[static_cast<size_t>(tgt.det_index)]
                                    : nullptr;
        const double t_track = pt::now_s();

        // ---- hands of the followed person
        if (!has || !a.hands) {
            hands.clear();
        } else {
            const float fw = tgt.x2 - tgt.x1, fcx = (tgt.x1 + tgt.x2) / 2, fcy = (tgt.y1 + tgt.y2) / 2;
            std::vector<HandTrack> kept;
            for (size_t i = 0; i < n_followed; ++i) {
                HandTrack h = hands[i];
                pt::decode_hand(hand_out[i], h.roi, hand_size, h.lm);
                if (h.lm.presence < a.hand_keep) continue;
                const float hx = h.lm.x[9], hy = h.lm.y[9];  // middle-finger base
                if (std::hypot(hx - fcx, hy - fcy) > a.region_width * fw) continue;  // drifted away from the person
                kept.push_back(h);
            }
            if (kept.size() == 2) {  // both crops on the same hand: keep the surer one
                const auto& r0 = kept[0].roi;
                const auto& r1 = kept[1].roi;
                if (std::hypot(r0.cx - r1.cx, r0.cy - r1.cy) < 0.5f * std::min(r0.size, r1.size))
                    kept.erase(kept.begin() + (kept[0].lm.presence >= kept[1].lm.presence ? 1 : 0));
            }
            hands.swap(kept);

            if (hands.size() < 2 && (hands.empty() || frames % a.search_every == 0)) {
                ++palm_searches;
                int rx, ry, rw, rh;
                pt::hand_search_region(tgt.x1, tgt.y1, tgt.x2, tgt.y2, W, H, a.region_width, a.region_above,
                                       a.region_below, rx, ry, rw, rh);
                const pt::Letterbox lb_palm = pt::make_letterbox(rx, ry, rw, rh, palm_size, palm_size);
                pt::letterbox_gpu(gf, lb_palm, mp_in, palm->input().dev, stream);
                palm->enqueue();
                palm->download();
                CUDA_CHECK(cudaStreamSynchronize(stream));
                palm_dec.decode(palm->output().host, lb_palm, palms);
                std::vector<pt::RotRect> fresh;
                for (const auto& p : palms) {
                    if (hands.size() + fresh.size() >= 2) break;
                    const float pcx = (p.x1 + p.x2) / 2, pcy = (p.y1 + p.y2) / 2;
                    bool known = false;
                    for (const auto& h : hands) known |= std::hypot(pcx - h.roi.cx, pcy - h.roi.cy) < 0.5f * h.roi.size;
                    if (!known) fresh.push_back(pt::hand_roi_from_palm(p));
                }
                for (size_t j = 0; j < fresh.size(); ++j) {
                    pt::rotcrop_gpu(gf, fresh[j], hand_size, mp_in, hand->input().dev, stream);
                    hand->enqueue();
                    hand->download(hand_out[2 + j]);
                }
                if (!fresh.empty()) CUDA_CHECK(cudaStreamSynchronize(stream));
                for (size_t j = 0; j < fresh.size(); ++j) {
                    HandTrack h;
                    pt::decode_hand(hand_out[2 + j], fresh[j], hand_size, h.lm);
                    if (h.lm.presence < a.hand_new) continue;
                    h.id = next_hand_id++;
                    h.side = h.lm.handedness;
                    hands.push_back(h);
                }
            }
            for (auto& h : hands) {  // gestures, side, next crop
                h.side = h.age == 0 ? h.lm.handedness : 0.8f * h.side + 0.2f * h.lm.handedness;
                ++h.age;
                const std::string& stable = h.smooth.update(pt::classify_gesture(h.lm));
                if (stable != h.reported) {
                    h.reported = stable;
                    if (!stable.empty()) {
                        ++gesture_counts[stable];
                        std::printf("[%8.2f s] target %d, %s hand: %s\n", fr.t_ready - t_start, tgt.id,
                                    person_side(h.side, a.mirrored), stable.c_str());
                        std::fflush(stdout);
                    }
                }
                h.roi = pt::hand_roi_from_landmarks(h.lm);
            }
        }
        const double t_done = pt::now_s();
        const double hand_ms = ms_hand + 1000.0 * (t_done - t_track);

        ++frames;
        has ? ++with_target : ++without_target;
        with_hand += !hands.empty();
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
            ov.fps = fps_ema;
            ov.latency_ms = total_ms;
            for (const auto& t : tracks) {
                const bool is_t = has && t.id == tgt.id;
                ov.boxes.push_back(pt::OverlayBox{t.x1, t.y1, t.x2, t.y2, is_t,
                                                  (is_t ? "TARGET id " : "id ") + std::to_string(t.id)});
            }
            if (tface) ov.points.assign(tface->pts, tface->pts + 10);
            for (const auto& h : hands) {
                pt::OverlayHand oh;
                std::copy(h.lm.x, h.lm.x + 21, oh.x);
                std::copy(h.lm.y, h.lm.y + 21, oh.y);
                oh.highlight = true;
                oh.label = std::string(person_side(h.side, a.mirrored)) + (h.reported.empty() ? "" : ": " + h.reported);
                ov.hands.push_back(oh);
            }
            ov.status = "faces " + std::to_string(faces.size()) + "  hands " + std::to_string(hands.size()) +
                        (has ? "  target id " + std::to_string(tgt.id) : std::string("  no target"));
            if (to_sink) sink->push(fr.data, fr.bytes, ov);
            if (to_view) view->submit(fr.data, fr.bytes, ov, t_done);
        }
        cam.release(fr);

        if (frames > a.warmup) {
            if (counted == 0) {
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
            s_hand.push_back(hand_ms);
            s_gpu_wall.push_back(1000.0 * (t_gpu - t_got));
            s_post.push_back(1000.0 * (t_post - t_gpu));
            s_track.push_back(1000.0 * (t_track - t_post));
            s_total.push_back(total_ms);
        }

        const char* evs = "-";
        switch (target.last_event()) {
            case pt::TargetSelector::Event::Acquired: evs = "acquired"; break;
            case pt::TargetSelector::Event::Kept: evs = "kept"; break;
            case pt::TargetSelector::Event::Relocked: evs = "relocked"; break;
            case pt::TargetSelector::Event::Lost: evs = "lost"; break;
            default: break;
        }
        std::string left, right;  // gesture per side ("-": a hand without a gesture)
        for (const auto& h : hands)
            (std::strcmp(person_side(h.side, a.mirrored), "left") == 0 ? left : right) =
                h.reported.empty() ? std::string("-") : h.reported;
        char row[640];
        const double ts = fr.t_ready - t_start;
        if (has)
            std::snprintf(row, sizeof(row),
                          "%ld,%.4f,%d,%.1f,%.1f,%.1f,%.1f,%.3f,%zu,%zu,%s,%s,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%s\n",
                          frames, ts, tgt.id, tgt.x1, tgt.y1, tgt.x2, tgt.y2, tgt.score, faces.size(), hands.size(),
                          left.c_str(), right.c_str(), 1000.0 * (t_got - t_req), ms_upload, ms_pre, ms_infer, hand_ms,
                          1000.0 * (t_post - t_gpu), 1000.0 * (t_track - t_post), total_ms, evs);
        else
            std::snprintf(row, sizeof(row), "%ld,%.4f,-1,,,,,,%zu,0,,,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%s\n",
                          frames, ts, faces.size(), 1000.0 * (t_got - t_req), ms_upload, ms_pre, ms_infer, hand_ms,
                          1000.0 * (t_post - t_gpu), 1000.0 * (t_track - t_post), total_ms, evs);
        if (csv) std::fputs(row, csv);
        if (!a.quiet && frames % a.print_every == 0) std::fputs(row, stdout);
        if (a.quiet && t_done - t_status >= 2.0) {
            t_status = t_done;
            std::fprintf(stderr, "[app] frame %ld  %.1f FPS  latency %.1f ms  faces %zu  target %d  hands %zu\n", frames,
                         fps_ema, total_ms, faces.size(), has ? tgt.id : -1, hands.size());
        }
        if (a.max_frames > 0 && frames >= a.max_frames) break;
        if (a.duration > 0 && t_done - t_start >= a.duration) break;
    }
    cam.stop();
    sink.reset();
    const unsigned long long view_frames = view ? view->frames_sent() : 0ULL;
    view.reset();
    if (csv) std::fclose(csv);

    // ---- summary
    const double span = t_last - t_first;
    const double fps = counted > 1 && span > 0 ? (counted - 1) / span : 0.0;
    const pt::Stat st_wait = pt::summarize(s_wait), st_up = pt::summarize(s_upload), st_pre = pt::summarize(s_pre),
                   st_inf = pt::summarize(s_infer), st_hand = pt::summarize(s_hand), st_gw = pt::summarize(s_gpu_wall),
                   st_post = pt::summarize(s_post), st_trk = pt::summarize(s_track), st_tot = pt::summarize(s_total);
    const double rss = rss_mb();
    const unsigned long long n_read = cam.frames_read() - read_base, n_drop = cam.frames_dropped() - drop_base;

    std::string md;
    char buf[1024];
    auto add = [&](const char* fmt, auto... xs) {
        std::snprintf(buf, sizeof(buf), fmt, xs...);
        md += buf;
    };
    add("## face_gesture run%s%s\n\n", a.label.empty() ? "" : ": ", a.label.c_str());
    add("- faces `%s` (input %dx%d); hands %s; CUDA graphs %s, frame copy %s\n", a.face_engine.c_str(), face.input_w(),
        face.input_h(), hands_desc.c_str(), face.graph_active() ? "on" : "off", a.zero_copy ? "zero-copy" : "explicit");
    add("- source %s, frames %dx%d %s, 1 of every %d processed\n", cam.description().c_str(), W, H,
        cam.format() == pt::PixelFormat::NV12 ? "NV12" : "BGR", a.every);
    add("- frames processed %ld (statistics over %ld after %d warm-up), %.1f s\n", frames, counted, a.warmup,
        pt::now_s() - t_start);
    add("- **throughput %.2f FPS**; after warm-up: frames read %llu, dropped as stale %llu (camera rate %.2f FPS)\n", fps,
        n_read, n_drop, cam.capture_fps());
    add("- followed face present in %ld frames, absent in %ld; its ID changed %d times (re-locks %d, re-acquisitions "
        "%d); face IDs seen %zu\n",
        with_target, without_target, target.id_changes(), target.relocks(), target.reacquires(), ids_seen.size());
    add("- hands: frames with at least one hand %ld; palm searches %ld\n", with_hand, palm_searches);
    std::string gl;
    for (const auto& g : gesture_counts) gl += (gl.empty() ? "" : ", ") + g.first + " " + std::to_string(g.second);
    add("- gestures recognised: %s\n", gl.empty() ? "none" : gl.c_str());
    add("- process RSS %.0f MB; live view frames sent %llu\n\n", rss, view_frames);
    md += "| stage (ms) | mean | p50 | p95 | max |\n|---|---|---|---|---|\n";
    auto row = [&](const char* name, const pt::Stat& s) {
        add("| %s | %.3f | %.3f | %.3f | %.3f |\n", name, s.mean, s.p50, s.p95, s.max);
    };
    row("wait for frame", st_wait);
    row("upload (GPU)", st_up);
    row("face input (GPU)", st_pre);
    row("face detector (GPU)", st_inf);
    row("GPU batch wall (faces + followed hands)", st_gw);
    row("face decode (CPU)", st_post);
    row("face tracker + target (CPU)", st_trk);
    row("hands (GPU + CPU, incl. searches)", st_hand);
    row("**latency: frame read -> result**", st_tot);
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
            std::fprintf(f,
                         "{\"app\":\"face_gesture\",\"label\":\"%s\",\"engine\":\"%s\",\"source\":\"%s\",\"input\":\"%dx%d\",",
                         a.label.c_str(), a.face_engine.c_str(), a.source.c_str(), face.input_w(), face.input_h());
            std::fprintf(f, "\"head\":\"YuNet\",\"hands\":%s,\"cuda_graph\":%s,\"zero_copy\":%s,\"frames\":%ld,"
                         "\"counted\":%ld,\"fps\":%.3f,",
                         a.hands ? "true" : "false", face.graph_active() ? "true" : "false",
                         a.zero_copy ? "true" : "false", frames, counted, fps);
            std::fprintf(f, "\"frame_size\":\"%dx%d\",\"hw_resize\":false,\"every\":%d,\"dla\":-1,\"view_frames\":%llu,",
                         W, H, a.every, view_frames);
            std::fprintf(f, "\"camera_mode\":\"%dx%d@%d\",\"source_fps\":%.2f,\"camera_fps\":%.2f,\"start_unix\":%.3f,",
                         a.width, a.height, a.fps, cam.source_fps(), cam.capture_fps(), t_start_unix);
            std::fprintf(f, "\"frames_read\":%llu,\"dropped\":%llu,\"gaps\":%llu,\"with_target\":%ld,\"without_target\":%ld,",
                         n_read, n_drop, static_cast<unsigned long long>(gaps), with_target, without_target);
            std::fprintf(f, "\"id_changes\":%d,\"relocks\":%d,\"reacquires\":%d,\"ids_seen\":%zu,\"rss_mb\":%.1f,",
                         target.id_changes(), target.relocks(), target.reacquires(), ids_seen.size(), rss);
            std::fprintf(f, "\"frames_with_hand\":%ld,\"palm_searches\":%ld,\"gestures\":{", with_hand, palm_searches);
            bool first = true;
            for (const auto& g : gesture_counts) {
                std::fprintf(f, "%s\"%s\":%d", first ? "" : ",", g.first.c_str(), g.second);
                first = false;
            }
            std::fprintf(f, "},");
            js("wait", st_wait);
            js("upload", st_up);
            js("pre", st_pre);
            js("infer", st_inf);
            js("hand", st_hand);
            js("gpu_wall", st_gw);
            js("post", st_post);
            js("track", st_trk);
            js("total", st_tot);
            std::fprintf(f, "\"ok\":true}\n");
            std::fclose(f);
        }
    }
    for (auto& e : ev) cudaEventDestroy(e);
    for (auto& p : hand_out) cudaFreeHost(p);
    if (frame_dev) cudaFree(frame_dev);
    hand.reset();
    palm.reset();
    cudaStreamDestroy(stream);
    return 0;
}
