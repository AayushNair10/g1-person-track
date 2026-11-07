#include "gestures.hpp"

#include <cmath>

namespace pt {

namespace {

struct V3 {
    float x, y, z;
};

V3 world(const HandLandmarks& h, int i) { return {h.wx[i], h.wy[i], h.wz[i]}; }
V3 sub(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
float len(V3 a) { return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z); }
float dist(V3 a, V3 b) { return len(sub(a, b)); }

// angle between the segments a->b and b->c, degrees (0 = straight on)
float bend(V3 a, V3 b, V3 c) {
    const V3 u = sub(b, a), v = sub(c, b);
    const float d = len(u) * len(v);
    if (d <= 0.f) return 0.f;
    float cs = (u.x * v.x + u.y * v.y + u.z * v.z) / d;
    cs = cs > 1.f ? 1.f : (cs < -1.f ? -1.f : cs);
    return std::acos(cs) * 180.f / kPi;
}

const char* direction(float dx, float dy, float ratio) {  // image vector, y down
    const float ax = std::fabs(dx), ay = std::fabs(dy), n = std::sqrt(dx * dx + dy * dy);
    if (n <= 0.f) return nullptr;
    if (ay >= ax && ay >= ratio * n) return dy < 0 ? "up" : "down";
    if (ax > ay && ax >= ratio * n) return dx < 0 ? "left" : "right";
    return nullptr;
}

}  // namespace

HandShape hand_shape(const HandLandmarks& h, const GestureConfig& c) {
    HandShape s{};
    const V3 wrist = world(h, 0);
    s.palm = dist(wrist, world(h, 9));
    const float palm = s.palm > 1e-4f ? s.palm : 1e-4f;
    // fingers: base (MCP) b, middle joint p, upper joint d, tip t
    for (int k = 1; k < 5; ++k) {
        const int b = 1 + 4 * k, p = b + 1, d = b + 2, t = b + 3;
        const float bent = bend(world(h, b), world(h, p), world(h, d)) + bend(world(h, p), world(h, d), world(h, t));
        const float tip = dist(world(h, t), wrist), mid = dist(world(h, p), wrist), base = dist(world(h, b), wrist);
        if (bent < c.straight_bend_deg && tip > mid)
            s.f[k] = Finger::Straight;
        else if (bent > c.curled_bend_deg || tip < base * 1.05f)
            s.f[k] = Finger::Curled;
        else
            s.f[k] = Finger::Half;
    }
    // thumb: joints 1 (CMC), 2 (MCP), 3 (IP), 4 (tip)
    const float tb = bend(world(h, 1), world(h, 2), world(h, 3)) + bend(world(h, 2), world(h, 3), world(h, 4));
    const float out = dist(world(h, 4), world(h, 5)) / palm;
    if (tb < c.thumb_straight_deg && out > c.thumb_out)
        s.f[0] = Finger::Straight;
    else if (out < 0.5f * c.thumb_out)
        s.f[0] = Finger::Curled;
    else
        s.f[0] = Finger::Half;
    s.touch = dist(world(h, 4), world(h, 8)) / palm;
    return s;
}

std::string classify_gesture(const HandLandmarks& h, const GestureConfig& c) {
    const HandShape s = hand_shape(h, c);
    const bool T = s.f[0] == Finger::Straight, I = s.f[1] == Finger::Straight, M = s.f[2] == Finger::Straight,
               R = s.f[3] == Finger::Straight, L = s.f[4] == Finger::Straight;
    const bool Tc = s.f[0] != Finger::Straight;  // thumb not out
    const bool Ic = s.f[1] == Finger::Curled, Mc = s.f[2] == Finger::Curled, Rc = s.f[3] == Finger::Curled,
               Lc = s.f[4] == Finger::Curled;

    if (s.touch < c.touch) {  // thumb and index tips together
        if (M && R && L) return "OK";
        if (!(M && R)) return "Pinch";
    }
    if (T && I && M && R && L) return "Open palm";
    if (Ic && Mc && Rc && Lc) {
        if (T) {
            const char* d = direction(h.x[4] - h.x[2], h.y[4] - h.y[2], c.dir_ratio);  // thumb base -> tip
            if (d && d[0] == 'u') return "Thumbs up";
            if (d && d[0] == 'd') return "Thumbs down";
            return "";
        }
        return "Fist";
    }
    if (T && Ic && Mc && Rc && L) return "Call me";
    if (I && Mc && Rc && L) return T ? "I love you" : "Rock";
    if (I && M && R && Lc && Tc) return "Three";
    if (I && M && Rc && Lc) return "Victory";
    if (I && Mc && Rc && Lc) {
        const char* d = direction(h.x[8] - h.x[5], h.y[8] - h.y[5], c.dir_ratio);  // index base -> tip
        if (d) return std::string("Pointing ") + d;
    }
    return "";
}

const std::string& GestureSmoother::update(const std::string& label) {
    hist_.push_back(label);
    while (static_cast<int>(hist_.size()) > window_) hist_.pop_front();
    std::string best;
    int best_n = 0;
    for (const auto& a : hist_) {
        int n = 0;
        for (const auto& b : hist_) n += a == b;
        if (n > best_n) {
            best_n = n;
            best = a;
        }
    }
    stable_ = best_n >= need_ ? best : std::string();
    return stable_;
}

}  // namespace pt
