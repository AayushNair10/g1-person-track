// YuNet face detections from models/yunet_<W>x<H>.onnx (scripts/export_face_hand.py). The model's output is
// already decoded inside TensorRT: the best 100 candidates as x1 y1 x2 y2 score + 5 points (x, y) in input
// pixels (points: right eye, left eye, nose tip, right and left mouth corner, as YuNet orders them).
// Here: score threshold, NMS (greedy, IoU) and the way back to frame pixels. Plain C++ (desktop-testable).
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "preprocess_math.hpp"

namespace pt {

struct Face {
    float x1, y1, x2, y2;  // frame pixels
    float score;
    float pts[10];         // 5 points (x, y), frame pixels
};

inline float iou(float ax1, float ay1, float ax2, float ay2, float bx1, float by1, float bx2, float by2) {
    const float iw = std::max(0.f, std::min(ax2, bx2) - std::max(ax1, bx1));
    const float ih = std::max(0.f, std::min(ay2, by2) - std::max(ay1, by1));
    const float inter = iw * ih;
    const float u = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
    return u > 0.f ? inter / u : 0.f;
}

class FaceDecoder {
public:
    static constexpr int kRows = 100, kCols = 15;

    // lb: how the frame was put into the model input; frame_w/h: to clip the boxes.
    FaceDecoder(const Letterbox& lb, int frame_w, int frame_h, float conf = 0.6f, float nms_iou = 0.3f)
        : lb_(lb), fw_(static_cast<float>(frame_w)), fh_(static_cast<float>(frame_h)), conf_(conf), nms_(nms_iou) {}

    void decode(const float* out, std::vector<Face>& faces) const {
        faces.clear();
        // rows are sorted by score (TopK), so the greedy NMS can go in order
        for (int i = 0; i < kRows; ++i) {
            const float* r = out + i * kCols;
            if (!(r[4] >= conf_)) break;
            bool keep = true;
            for (const auto& k : kept_in_)
                if (iou(r[0], r[1], r[2], r[3], k[0], k[1], k[2], k[3]) > nms_) {
                    keep = false;
                    break;
                }
            if (!keep) continue;
            kept_in_.push_back({r[0], r[1], r[2], r[3]});
            Face f;
            f.x1 = clampx(lb_to_frame_x(lb_, r[0]));
            f.y1 = clampy(lb_to_frame_y(lb_, r[1]));
            f.x2 = clampx(lb_to_frame_x(lb_, r[2]));
            f.y2 = clampy(lb_to_frame_y(lb_, r[3]));
            f.score = r[4];
            for (int p = 0; p < 5; ++p) {
                f.pts[2 * p] = lb_to_frame_x(lb_, r[5 + 2 * p]);
                f.pts[2 * p + 1] = lb_to_frame_y(lb_, r[6 + 2 * p]);
            }
            faces.push_back(f);
        }
        kept_in_.clear();
    }

private:
    struct Box {
        float v[4];
        float operator[](int i) const { return v[i]; }
    };
    float clampx(float x) const { return std::min(std::max(x, 0.f), fw_); }
    float clampy(float y) const { return std::min(std::max(y, 0.f), fh_); }

    Letterbox lb_;
    float fw_, fh_, conf_, nms_;
    mutable std::vector<Box> kept_in_;
};

}  // namespace pt
