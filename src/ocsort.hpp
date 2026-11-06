// OC-SORT multi-object tracker in plain C++17 (no Eigen, no OpenCV).
//
// A line-by-line port of the Ultralytics 8.4 OC-SORT (ultralytics/trackers/oc_sort.py on top of
// byte_tracker.py), the tracker the Mac benchmark ran (ocsort.yaml defaults below):
//   - Kalman filter on (cx, cy, aspect, h) + velocities: 8-state constant velocity, 4-d measurement,
//     hand-written 8x8 / 4x4 matrix math (same noise model as Ultralytics' KalmanFilterXYAH)
//   - cost = 1 - IoU * score  +  inertia * OCM (angle between track motion and track->detection)
//   - optimal assignment (Hungarian / Jonker-Volgenant style) with a cost limit, like lap.lapjv
//   - OCR: second pass on unmatched tracks using their last real observation box
//   - ORU: when a lost track is recovered by OCR, the Kalman state is re-run over virtual
//     observations interpolated across the gap
// Same stages, same order, same float/double choices, so IDs match the Python tracker frame for frame
// (tests/desktop/test_tracker.cpp checks this on the benchmark clips).
#pragma once

#include <array>
#include <memory>
#include <vector>

namespace pt {

// One detection in frame pixels.
struct Det {
    float x1, y1, x2, y2;
    float score;
    int cls = 0;
};

// One tracked object, as Ultralytics returns it: box from the Kalman state, plus the detection it
// matched this frame (det_index into the update() input, -1 never happens for returned tracks).
struct TrackOut {
    int id;
    float x1, y1, x2, y2;
    float score;
    int cls;
    int det_index;
};

struct OCSortConfig {
    float track_high_thresh = 0.25f;  // detections below this are ignored (no BYTE second pass)
    float new_track_thresh = 0.25f;   // min score to start a track
    int track_buffer = 30;            // frames a lost track is kept
    float match_thresh = 0.8f;        // max cost for a match
    bool fuse_score = true;           // cost uses IoU * score
    int delta_t = 3;                  // frames back for the motion direction
    float inertia = 0.2f;             // weight of the direction cost
};

class OCSort {
public:
    explicit OCSort(const OCSortConfig& cfg = OCSortConfig());
    ~OCSort();
    OCSort(const OCSort&) = delete;
    OCSort& operator=(const OCSort&) = delete;

    // Feeds one frame of detections; returns the confirmed, currently tracked objects.
    const std::vector<TrackOut>& update(const std::vector<Det>& dets);
    void reset();
    int frame_id() const { return frame_id_; }
    const OCSortConfig& config() const { return cfg_; }

    struct Track;  // internal
    using TrackPtr = std::shared_ptr<Track>;

private:
    OCSortConfig cfg_;
    int frame_id_ = 0;
    int next_id_ = 0;
    std::vector<TrackPtr> tracked_, lost_;
    std::vector<TrackOut> out_;
};

// Optimal assignment with a cost limit, like lap.lapjv(cost, extend_cost=True, cost_limit=thresh):
// pairs costing more than `thresh` are never chosen. Returns row->col (-1 = unmatched) and col->row.
// Exposed for the tests.
void linear_assignment(const std::vector<double>& cost, int rows, int cols, double thresh,
                       std::vector<int>& row_to_col, std::vector<int>& col_to_row);

}  // namespace pt
