// Hands: palm detections (models/palm_192.onnx), the rotated hand crop, the 21 hand landmarks
// (models/handpose_224.onnx) and the crop for the next frame. Plain C++ (desktop-testable).
//
// Crop geometry follows MediaPipe's published hand-tracking recipe: from a palm, a square 2.6x the palm box,
// shifted half a palm towards the fingers and turned so the wrist -> middle-finger direction points up; from
// landmarks, a square 2x the hand's extent (in that turned frame), shifted 0.1 towards the fingers.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "face_detector.hpp"
#include "preprocess_math.hpp"

namespace pt {

constexpr float kPi = 3.14159265358979323846f;

struct Palm {
    float x1, y1, x2, y2;  // frame pixels
    float score;
    float pts[14];         // 7 points (x, y): 0 wrist centre, 1 index base, 2 middle-finger base, ...
};

// Palm model output (1x32x21 rows, best first): score logit, 18 regressors, anchor x y.
// lb: how the search region was put into the 192x192 input.
class PalmDecoder {
public:
    static constexpr int kRows = 32, kCols = 21, kInput = 192;

    explicit PalmDecoder(float conf = 0.5f, float nms_iou = 0.3f) : conf_(conf), nms_(nms_iou) {}

    void decode(const float* out, const Letterbox& lb, std::vector<Palm>& palms) const {
        palms.clear();
        for (int i = 0; i < kRows; ++i) {
            const float* r = out + i * kCols;
            const float score = 1.f / (1.f + std::exp(-r[0]));
            if (!(score >= conf_)) break;  // rows sorted by score
            const float* g = r + 1;
            const float ax = r[19] * kInput, ay = r[20] * kInput;  // anchor, input pixels
            const float cx = ax + g[0], cy = ay + g[1], w = g[2], h = g[3];
            Palm p;
            p.x1 = lb_to_frame_x(lb, cx - w / 2);
            p.y1 = lb_to_frame_y(lb, cy - h / 2);
            p.x2 = lb_to_frame_x(lb, cx + w / 2);
            p.y2 = lb_to_frame_y(lb, cy + h / 2);
            p.score = score;
            for (int k = 0; k < 7; ++k) {
                p.pts[2 * k] = lb_to_frame_x(lb, ax + g[4 + 2 * k]);
                p.pts[2 * k + 1] = lb_to_frame_y(lb, ay + g[5 + 2 * k]);
            }
            bool keep = true;
            for (const auto& q : palms)
                if (iou(p.x1, p.y1, p.x2, p.y2, q.x1, q.y1, q.x2, q.y2) > nms_) {
                    keep = false;
                    break;
                }
            if (keep) palms.push_back(p);
        }
    }

private:
    float conf_, nms_;
};

// Angle that turns the square's up onto the direction from (x0, y0) to (x1, y1) (see RotRect).
inline float up_angle(float x0, float y0, float x1, float y1) { return std::atan2(x1 - x0, -(y1 - y0)); }

inline RotRect hand_roi_from_palm(const Palm& p) {
    const float a = up_angle(p.pts[0], p.pts[1], p.pts[4], p.pts[5]);  // wrist centre -> middle-finger base
    const float w = p.x2 - p.x1, h = p.y2 - p.y1;
    RotRect r{(p.x1 + p.x2) / 2, (p.y1 + p.y2) / 2, 2.6f * std::max(w, h), a};
    rot_to_frame(r, 0.f, -0.5f * h, r.cx, r.cy);  // half a palm towards the fingers
    return r;
}

struct HandLandmarks {
    float x[21], y[21], z[21];     // frame pixels (z: relative depth, frame-pixel scale)
    float wx[21], wy[21], wz[21];  // world landmarks (metres, hand-centred)
    float presence = 0.f;          // hand present in the crop (0..1)
    float handedness = 0.f;        // model output: > 0.5 = "right" (see HandTracker for the camera's view)
};

// Hand model output (1x128): 21 x (x y z) in input pixels, presence, handedness, 21 x (x y z) world.
inline void decode_hand(const float* out, const RotRect& roi, int input, HandLandmarks& h) {
    for (int i = 0; i < 21; ++i) {
        crop_to_frame(roi, input, out[3 * i], out[3 * i + 1], h.x[i], h.y[i]);
        h.z[i] = out[3 * i + 2] * roi.size / input;
        h.wx[i] = out[65 + 3 * i];
        h.wy[i] = out[66 + 3 * i];
        h.wz[i] = out[67 + 3 * i];
    }
    h.presence = out[63];
    h.handedness = out[64];
}

// Next frame's crop from this frame's landmarks.
inline RotRect hand_roi_from_landmarks(const HandLandmarks& h) {
    RotRect r{h.x[0], h.y[0], 1.f, up_angle(h.x[0], h.y[0], h.x[9], h.y[9])};  // wrist -> middle-finger base
    float lo_x = 1e9f, lo_y = 1e9f, hi_x = -1e9f, hi_y = -1e9f;
    for (int i = 0; i < 21; ++i) {
        float dx, dy;
        frame_to_rot(r, h.x[i], h.y[i], dx, dy);
        lo_x = std::min(lo_x, dx);
        hi_x = std::max(hi_x, dx);
        lo_y = std::min(lo_y, dy);
        hi_y = std::max(hi_y, dy);
    }
    const float w = hi_x - lo_x, ht = hi_y - lo_y;
    float cx, cy;
    rot_to_frame(r, (lo_x + hi_x) / 2, (lo_y + hi_y) / 2 - 0.1f * ht, cx, cy);
    return RotRect{cx, cy, 2.f * std::max(w, ht), r.angle};
}

inline void hand_box(const HandLandmarks& h, float& x1, float& y1, float& x2, float& y2) {
    x1 = *std::min_element(h.x, h.x + 21);
    x2 = *std::max_element(h.x, h.x + 21);
    y1 = *std::min_element(h.y, h.y + 21);
    y2 = *std::max_element(h.y, h.y + 21);
}

// Where to look for the followed person's hands: a region around their face (frame pixels, clipped).
inline void hand_search_region(float fx1, float fy1, float fx2, float fy2, int frame_w, int frame_h, float width_faces,
                               float above_faces, float below_faces, int& x0, int& y0, int& w, int& h) {
    const float fw = fx2 - fx1, fh = fy2 - fy1, cx = (fx1 + fx2) / 2;
    const float rx1 = std::max(0.f, cx - width_faces * fw / 2), rx2 = std::min<float>(frame_w, cx + width_faces * fw / 2);
    const float ry1 = std::max(0.f, fy1 - above_faces * fh), ry2 = std::min<float>(frame_h, fy2 + below_faces * fh);
    x0 = static_cast<int>(rx1);
    y0 = static_cast<int>(ry1);
    w = std::max(1, static_cast<int>(rx2) - x0);
    h = std::max(1, static_cast<int>(ry2) - y0);
}

}  // namespace pt
