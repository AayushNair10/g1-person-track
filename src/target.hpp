// Chooses the one person to follow among the tracker's outputs, and keeps it.
//   start      : the largest tracked person becomes the target
//   each frame : the track with the target's ID is the target
//   ID missing : re-lock to the track with the highest IoU to the target's last box
//                (if that IoU >= relock_min_iou); this keeps the same physical person when the
//                tracker hands them a new ID
//   still none : after `reacquire_after` frames with no target (the tracker has dropped the old
//                track by then: track_buffer = 30), start over with the largest person
#pragma once

#include <algorithm>
#include <vector>

#include "ocsort.hpp"

namespace pt {

struct TargetConfig {
    float relock_min_iou = 0.3f;
    int reacquire_after = 30;
};

class TargetSelector {
public:
    explicit TargetSelector(const TargetConfig& cfg = TargetConfig()) : cfg_(cfg) {}

    enum class Event { None, Acquired, Kept, Relocked, Lost };

    // Returns true and fills `out` when there is a target this frame.
    bool update(const std::vector<TrackOut>& tracks, TrackOut& out) {
        event_ = Event::None;
        if (id_ >= 0) {
            for (const auto& t : tracks)
                if (t.id == id_) return keep(t, out, Event::Kept);
            ++missing_;
            const TrackOut* best = nullptr;
            float best_iou = 0.f;
            for (const auto& t : tracks) {
                const float v = iou(t);
                if (v > best_iou) {
                    best_iou = v;
                    best = &t;
                }
            }
            if (best && best_iou >= cfg_.relock_min_iou) {
                id_ = best->id;
                ++relocks_;
                return keep(*best, out, Event::Relocked);
            }
            if (missing_ <= cfg_.reacquire_after) {
                event_ = Event::Lost;
                return false;
            }
            id_ = -1;  // give up on the old target
        }
        if (tracks.empty()) return false;
        const TrackOut* big = &tracks[0];
        for (const auto& t : tracks)
            if (area(t) > area(*big)) big = &t;
        if (had_target_) ++reacquires_;
        id_ = big->id;
        had_target_ = true;
        return keep(*big, out, Event::Acquired);
    }

    int target_id() const { return id_; }
    int relocks() const { return relocks_; }        // ID changed, same place (IoU re-lock)
    int reacquires() const { return reacquires_; }  // target lost for good, new largest person
    int id_changes() const { return relocks_ + reacquires_; }
    Event last_event() const { return event_; }

private:
    static float area(const TrackOut& t) { return std::max(0.f, t.x2 - t.x1) * std::max(0.f, t.y2 - t.y1); }

    float iou(const TrackOut& t) const {
        const float iw = std::max(0.f, std::min(last_[2], t.x2) - std::max(last_[0], t.x1));
        const float ih = std::max(0.f, std::min(last_[3], t.y2) - std::max(last_[1], t.y1));
        const float inter = iw * ih;
        const float uni = (last_[2] - last_[0]) * (last_[3] - last_[1]) + area(t) - inter;
        return uni > 0.f ? inter / uni : 0.f;
    }

    bool keep(const TrackOut& t, TrackOut& out, Event e) {
        last_[0] = t.x1;
        last_[1] = t.y1;
        last_[2] = t.x2;
        last_[3] = t.y2;
        missing_ = 0;
        event_ = e;
        out = t;
        return true;
    }

    TargetConfig cfg_;
    int id_ = -1;
    float last_[4] = {0, 0, 0, 0};
    int missing_ = 0;
    int relocks_ = 0, reacquires_ = 0;
    bool had_target_ = false;
    Event event_ = Event::None;
};

}  // namespace pt
