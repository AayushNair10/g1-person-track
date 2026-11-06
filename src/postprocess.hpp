// Model output -> person detections in camera-frame pixels (no OpenCV, no CUDA).
//
//   NMS-free head (YOLO26 exported with nms=False): output (1, K, 6) = x1 y1 x2 y2 score class, in input
//     pixels, sorted by score. Keep class == person and score > conf (Ultralytics uses '>'), no NMS.
//   Raw head (YOLO26 nms=None, YOLO11, YOLOv8): output (1, 4 + nc, N) = cx cy w h, then nc class scores.
//     Like Ultralytics non_max_suppression(classes=[0]): a box counts only if person is its best class
//     and that score > conf; then greedy NMS (IoU > iou_thr suppresses), highest score first.
// Boxes then go back through the letterbox (Ultralytics scale_boxes + clip_boxes).
#pragma once

#include <algorithm>
#include <cstdio>
#include <vector>

#include "ocsort.hpp"           // Det
#include "preprocess_math.hpp"  // Letterbox

namespace pt {

enum class Head { NmsFree, Raw };

struct OutputLayout {
    Head head;
    int rows;     // NMS-free: K (300) detections; raw: 4 + number of classes
    int anchors;  // raw: N anchors; NMS-free: 6
};

// Picks the decoder from the output shape. dims: e.g. {1, 300, 6} or {1, 84, 5040}.
inline bool layout_from_dims(const int* dims, int nb, OutputLayout& out) {
    if (nb == 3 && dims[0] == 1 && dims[2] == 6) {
        out = {Head::NmsFree, dims[1], 6};
        return true;
    }
    if (nb == 3 && dims[0] == 1 && dims[1] > 4 && dims[2] > dims[1]) {
        out = {Head::Raw, dims[1], dims[2]};
        return true;
    }
    return false;
}

// Input-pixel box -> frame-pixel box, clipped to the frame.
inline void unletterbox(const Letterbox& lb, Det& d) {
    const float w = static_cast<float>(lb.src_w), h = static_cast<float>(lb.src_h);
    d.x1 = std::min(std::max((d.x1 - lb.left) / lb.gain_x, 0.f), w);
    d.x2 = std::min(std::max((d.x2 - lb.left) / lb.gain_x, 0.f), w);
    d.y1 = std::min(std::max((d.y1 - lb.top) / lb.gain_y, 0.f), h);
    d.y2 = std::min(std::max((d.y2 - lb.top) / lb.gain_y, 0.f), h);
}

inline float box_iou(const Det& a, const Det& b) {  // torchvision.ops.nms IoU (float32, no eps)
    const float iw = std::max(0.f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    const float ih = std::max(0.f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    const float inter = iw * ih;
    const float uni = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return uni > 0.f ? inter / uni : 0.f;
}

class Decoder {
public:
    Decoder(const OutputLayout& layout, const Letterbox& lb, float conf, float iou_thr = 0.7f, int cls = 0,
            int max_det = 300)
        : L_(layout), lb_(lb), conf_(conf), iou_(iou_thr), cls_(cls), max_det_(max_det) {}

    // out: the model's whole output tensor (host). Fills dets in frame pixels, best score first.
    void decode(const float* out, std::vector<Det>& dets) {
        dets.clear();
        if (L_.head == Head::NmsFree) {
            for (int i = 0; i < L_.rows; ++i) {
                const float* p = out + i * 6;
                if (!(p[4] > conf_) || p[5] != static_cast<float>(cls_)) continue;
                Det d{p[0], p[1], p[2], p[3], p[4], cls_};
                unletterbox(lb_, d);
                dets.push_back(d);
                if (static_cast<int>(dets.size()) >= max_det_) break;
            }
            return;
        }
        // raw head: candidates whose best class is `cls_` with score > conf
        const int n = L_.anchors, nc = L_.rows - 4;
        const float* cls_row = out + (4 + cls_) * n;
        cand_.clear();
        for (int a = 0; a < n; ++a) {
            const float s = cls_row[a];
            if (!(s > conf_)) continue;
            bool best = true;
            for (int c = 0; c < nc && best; ++c) {
                if (c == cls_) continue;
                const float o = out[(4 + c) * n + a];
                // torch max() returns the first index on ties, so an earlier class wins a tie
                if (o > s || (o == s && c < cls_)) best = false;
            }
            if (!best) continue;
            const float cx = out[a], cy = out[n + a], w = out[2 * n + a], h = out[3 * n + a];
            cand_.push_back({cx - w / 2.f, cy - h / 2.f, cx + w / 2.f, cy + h / 2.f, s, cls_});
        }
        std::stable_sort(cand_.begin(), cand_.end(), [](const Det& x, const Det& y) { return x.score > y.score; });
        removed_.assign(cand_.size(), 0);
        for (size_t i = 0; i < cand_.size() && static_cast<int>(dets.size()) < max_det_; ++i) {
            if (removed_[i]) continue;
            dets.push_back(cand_[i]);
            for (size_t j = i + 1; j < cand_.size(); ++j)
                if (!removed_[j] && box_iou(cand_[i], cand_[j]) > iou_) removed_[j] = 1;
        }
        for (auto& d : dets) unletterbox(lb_, d);
    }

    const OutputLayout& layout() const { return L_; }

private:
    OutputLayout L_;
    Letterbox lb_;
    float conf_, iou_;
    int cls_, max_det_;
    std::vector<Det> cand_;
    std::vector<char> removed_;
};

}  // namespace pt
