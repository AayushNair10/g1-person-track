// OC-SORT, ported from Ultralytics 8.4 (trackers/oc_sort.py, byte_tracker.py, utils/kalman_filter.py,
// utils/matching.py, utils/stracks.py). Comments name the Python function each block follows.
#include "ocsort.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <utility>

namespace pt {

namespace {

// ---------------------------------------------------------------- Kalman filter (KalmanFilterXYAH)
// State x = (cx, cy, a, h, vcx, vcy, va, vh), measurement z = (cx, cy, a, h). F = I + dt on (i, i+4).
constexpr double kPi = 3.14159265358979323846;
constexpr double kStdPos = 1.0 / 20.0;
constexpr double kStdVel = 1.0 / 160.0;

struct Kalman {
    static void initiate(const float z_in[4], double mean[8], double cov[64]) {
        double z[4] = {z_in[0], z_in[1], z_in[2], z_in[3]};
        for (int i = 0; i < 4; ++i) {
            mean[i] = z[i];
            mean[4 + i] = 0.0;
        }
        const double h = z[3];
        const double std[8] = {2 * kStdPos * h, 2 * kStdPos * h, 1e-2, 2 * kStdPos * h,
                               10 * kStdVel * h, 10 * kStdVel * h, 1e-5, 10 * kStdVel * h};
        for (int i = 0; i < 64; ++i) cov[i] = 0.0;
        for (int i = 0; i < 8; ++i) cov[i * 8 + i] = std[i] * std[i];
    }

    // mean = F mean; cov = F cov F^T + Q (Q from the height before the step)
    static void predict(double mean[8], double cov[64]) {
        const double h = mean[3];
        const double std[8] = {kStdPos * h, kStdPos * h, 1e-2, kStdPos * h,
                               kStdVel * h, kStdVel * h, 1e-5, kStdVel * h};
        for (int i = 0; i < 4; ++i) mean[i] += mean[4 + i];
        // A = F cov: row i (< 4) += row i+4
        double a[64];
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 8; ++j) a[i * 8 + j] = cov[i * 8 + j] + (i < 4 ? cov[(i + 4) * 8 + j] : 0.0);
        // cov = A F^T: column j (< 4) += column j+4
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 8; ++j) cov[i * 8 + j] = a[i * 8 + j] + (j < 4 ? a[i * 8 + j + 4] : 0.0);
        for (int i = 0; i < 8; ++i) cov[i * 8 + i] += std[i] * std[i];
    }

    // Standard update with measurement noise R = diag(std(h)^2): K = P H^T S^-1, S = H P H^T + R.
    static void update(double mean[8], double cov[64], const float z_in[4]) {
        const double h = mean[3];
        const double r[4] = {kStdPos * h, kStdPos * h, 1e-1, kStdPos * h};
        double s[16];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) s[i * 4 + j] = cov[i * 8 + j] + (i == j ? r[i] * r[i] : 0.0);
        double si[16];
        invert4(s, si);
        // K (8x4) = P[:, :4] * S^-1
        double k[32];
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 4; ++j) {
                double acc = 0.0;
                for (int t = 0; t < 4; ++t) acc += cov[i * 8 + t] * si[t * 4 + j];
                k[i * 4 + j] = acc;
            }
        double innov[4];
        for (int i = 0; i < 4; ++i) innov[i] = static_cast<double>(z_in[i]) - mean[i];
        for (int i = 0; i < 8; ++i) {
            double acc = 0.0;
            for (int t = 0; t < 4; ++t) acc += k[i * 4 + t] * innov[t];
            mean[i] += acc;
        }
        // cov -= K S K^T
        double ks[32];
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 4; ++j) {
                double acc = 0.0;
                for (int t = 0; t < 4; ++t) acc += k[i * 4 + t] * s[t * 4 + j];
                ks[i * 4 + j] = acc;
            }
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 8; ++j) {
                double acc = 0.0;
                for (int t = 0; t < 4; ++t) acc += ks[i * 4 + t] * k[j * 4 + t];
                cov[i * 8 + j] -= acc;
            }
    }

    // 4x4 inverse by Gauss-Jordan with partial pivoting (S is symmetric positive definite).
    static void invert4(const double m_in[16], double inv[16]) {
        double a[4][8];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                a[i][j] = m_in[i * 4 + j];
                a[i][4 + j] = (i == j) ? 1.0 : 0.0;
            }
        for (int c = 0; c < 4; ++c) {
            int p = c;
            for (int r = c + 1; r < 4; ++r)
                if (std::fabs(a[r][c]) > std::fabs(a[p][c])) p = r;
            if (p != c)
                for (int j = 0; j < 8; ++j) std::swap(a[c][j], a[p][j]);
            const double d = a[c][c];
            for (int j = 0; j < 8; ++j) a[c][j] /= d;
            for (int r = 0; r < 4; ++r) {
                if (r == c) continue;
                const double f = a[r][c];
                if (f == 0.0) continue;
                for (int j = 0; j < 8; ++j) a[r][j] -= f * a[c][j];
            }
        }
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) inv[i * 4 + j] = a[i][4 + j];
    }
};

// ---------------------------------------------------------------- box helpers
using Box = std::array<float, 4>;  // x1 y1 x2 y2

// STrack.tlwh_to_xyah (float32 arithmetic, like the numpy float32 arrays)
void tlwh_to_xyah(const float tlwh[4], float z[4]) {
    z[0] = tlwh[0] + tlwh[2] / 2.f;
    z[1] = tlwh[1] + tlwh[3] / 2.f;
    z[2] = tlwh[2] / tlwh[3];
    z[3] = tlwh[3];
}

void xyxy_to_xyah(const Box& b, float z[4]) {  // xyxy2ltwh then tlwh_to_xyah
    const float tlwh[4] = {b[0], b[1], b[2] - b[0], b[3] - b[1]};
    tlwh_to_xyah(tlwh, z);
}

// bbox_ioa(iou=True) on float32 arrays, eps 1e-7
float iou(const Box& a, const Box& b) {
    const float iw = std::max(0.f, std::min(a[2], b[2]) - std::max(a[0], b[0]));
    const float ih = std::max(0.f, std::min(a[3], b[3]) - std::max(a[1], b[1]));
    const float inter = iw * ih;
    const float area = (b[2] - b[0]) * (b[3] - b[1]) + (a[2] - a[0]) * (a[3] - a[1]) - inter;
    return inter / (area + 1e-7f);
}

std::array<float, 2> center(const Box& b) { return {(b[0] + b[2]) / 2.f, (b[1] + b[3]) / 2.f}; }

enum class State { New, Tracked, Lost, Removed };

}  // namespace

// ---------------------------------------------------------------- one track (STrack + OCSortTrack)
struct OCSort::Track {
    float tlwh0[4];  // detection box (STrack._tlwh, float32)
    bool has_state = false;
    double mean[8], cov[64];
    bool is_activated = false;
    State state = State::New;
    float score = 0.f;
    int cls = 0;
    int idx = -1;
    int track_id = 0, frame_id = 0, start_frame = 0, tracklet_len = 0;
    // OC-SORT
    Box last_obs{-1.f, -1.f, -1.f, -1.f};
    std::vector<std::pair<int, Box>> obs;  // (frame, box), oldest first, at most delta_t + 2
    bool has_vel = false;
    float vel[2] = {0.f, 0.f};
    bool has_saved = false;
    double saved_mean[8], saved_cov[64];

    // STrack.tlwh / .xyxy: from the Kalman mean (float64) when there is one, else the detection box
    Box xyxy() const {
        if (!has_state) {
            Box b{tlwh0[0], tlwh0[1], tlwh0[2], tlwh0[3]};
            b[2] += b[0];
            b[3] += b[1];
            return b;
        }
        double t[4] = {mean[0], mean[1], mean[2], mean[3]};
        t[2] *= t[3];
        t[0] -= t[2] / 2;
        t[1] -= t[3] / 2;
        t[2] += t[0];
        t[3] += t[1];
        return {static_cast<float>(t[0]), static_cast<float>(t[1]), static_cast<float>(t[2]),
                static_cast<float>(t[3])};
    }
    void tlwh(float out[4]) const {  // only used on detections
        for (int i = 0; i < 4; ++i) out[i] = tlwh0[i];
    }

    void save_state() {
        std::copy(mean, mean + 8, saved_mean);
        std::copy(cov, cov + 64, saved_cov);
        has_saved = true;
    }

    // OCSortTrack._record_observation
    void record(const Box& b, int frame, int delta_t) {
        last_obs = b;
        auto it = std::find_if(obs.begin(), obs.end(), [&](const std::pair<int, Box>& o) { return o.first == frame; });
        if (it != obs.end()) {
            it->second = b;
        } else {
            obs.emplace_back(frame, b);
            std::sort(obs.begin(), obs.end(),
                      [](const std::pair<int, Box>& x, const std::pair<int, Box>& y) { return x.first < y.first; });
        }
        const size_t keep = static_cast<size_t>(delta_t + 2);
        if (obs.size() > keep) obs.erase(obs.begin(), obs.begin() + static_cast<long>(obs.size() - keep));
    }

    // OCSortTrack._compute_velocity
    void compute_velocity(int delta_t) {
        has_vel = false;
        if (obs.size() < 2) return;
        const int cur_frame = obs.back().first;
        const auto cc = center(obs.back().second);
        const Box* prev = nullptr;
        for (auto it = obs.rbegin(); it != obs.rend(); ++it)
            if (it->first < cur_frame - delta_t + 1) {
                prev = &it->second;
                break;
            }
        if (!prev) {
            if (obs.front().first == cur_frame) return;
            prev = &obs.front().second;
        }
        const auto pc = center(*prev);
        const float dx = cc[0] - pc[0], dy = cc[1] - pc[1];
        const float n = std::sqrt(dx * dx + dy * dy);
        has_vel = true;
        if (n < 1e-6f) {
            vel[0] = vel[1] = 0.f;
        } else {
            vel[0] = dx / n;
            vel[1] = dy / n;
        }
    }

    // STrack.activate + OCSortTrack.activate
    void activate(int frame, int new_id, int delta_t) {
        track_id = new_id;
        float z[4];
        tlwh_to_xyah(tlwh0, z);
        Kalman::initiate(z, mean, cov);
        has_state = true;
        tracklet_len = 0;
        state = State::Tracked;
        if (frame == 1) is_activated = true;
        frame_id = frame;
        start_frame = frame;
        record(xyxy(), frame, delta_t);
        save_state();
    }

    // OCSortTrack.update (STrack.update in the middle)
    void update(const Track& det, int frame, int delta_t) {
        record(det.xyxy(), frame, delta_t);
        frame_id = frame;
        ++tracklet_len;
        float t[4], z[4];
        det.tlwh(t);
        tlwh_to_xyah(t, z);
        Kalman::update(mean, cov, z);
        state = State::Tracked;
        is_activated = true;
        score = det.score;
        cls = det.cls;
        idx = det.idx;
        save_state();
        compute_velocity(delta_t);
    }

    // OCSortTrack.re_activate (new_id=False)
    void re_activate(const Track& det, int frame, int delta_t) {
        record(det.xyxy(), frame, delta_t);
        float t[4], z[4];
        det.tlwh(t);
        tlwh_to_xyah(t, z);
        Kalman::update(mean, cov, z);
        tracklet_len = 0;
        state = State::Tracked;
        is_activated = true;
        frame_id = frame;
        score = det.score;
        cls = det.cls;
        idx = det.idx;
        save_state();
        compute_velocity(delta_t);
    }

    // OCSortTrack.apply_oru: re-run the filter from the last observation over interpolated boxes
    void apply_oru(const Box& new_obs, int cur_frame) {
        if (!has_saved || obs.empty()) return;
        const int last_frame = obs.back().first;
        const int gap = cur_frame - last_frame;
        if (gap <= 1) return;
        std::copy(saved_mean, saved_mean + 8, mean);
        std::copy(saved_cov, saved_cov + 64, cov);
        const Box last = obs.back().second;
        for (int t = 1; t < gap; ++t) {
            const double alpha = static_cast<double>(t) / gap;
            Box v;
            for (int i = 0; i < 4; ++i)
                v[i] = static_cast<float>(1.0 - alpha) * last[i] + static_cast<float>(alpha) * new_obs[i];
            float z[4];
            xyxy_to_xyah(v, z);
            Kalman::predict(mean, cov);
            Kalman::update(mean, cov, z);
        }
        Kalman::predict(mean, cov);
    }
};

// ---------------------------------------------------------------- assignment (lap.lapjv, cost_limit)
void linear_assignment(const std::vector<double>& cost, int rows, int cols, double thresh,
                       std::vector<int>& row_to_col, std::vector<int>& col_to_row) {
    row_to_col.assign(static_cast<size_t>(rows), -1);
    col_to_row.assign(static_cast<size_t>(cols), -1);
    if (rows == 0 || cols == 0) return;
    // lapjv's extension: (rows+cols) square; real block = cost, dummy blocks = thresh/2, dummy-dummy = 0
    const int n = rows + cols;
    std::vector<double> a(static_cast<size_t>(n) * n, thresh / 2.0);
    for (int i = rows; i < n; ++i)
        for (int j = cols; j < n; ++j) a[static_cast<size_t>(i) * n + j] = 0.0;
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) a[static_cast<size_t>(i) * n + j] = cost[static_cast<size_t>(i) * cols + j];
    // Hungarian algorithm (shortest augmenting paths with potentials), 1-indexed.
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> u(n + 1, 0.0), v(n + 1, 0.0), minv(n + 1);
    std::vector<int> p(n + 1, 0), way(n + 1, 0);
    std::vector<char> used(n + 1);
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::fill(minv.begin(), minv.end(), inf);
        std::fill(used.begin(), used.end(), 0);
        do {
            used[j0] = 1;
            const int i0 = p[j0];
            double delta = inf;
            int j1 = 0;
            for (int j = 1; j <= n; ++j) {
                if (used[j]) continue;
                const double cur = a[static_cast<size_t>(i0 - 1) * n + (j - 1)] - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    for (int j = 1; j <= n; ++j) {
        const int i = p[j] - 1, c = j - 1;
        if (i < rows && c < cols) {
            row_to_col[i] = c;
            col_to_row[c] = i;
        }
    }
}

namespace {

using TrackPtr = OCSort::TrackPtr;
using Tracks = std::vector<TrackPtr>;

struct Assign {
    std::vector<std::pair<int, int>> matches;  // (row, col), by row
    std::vector<int> u_rows, u_cols;           // ascending
};

// matching.linear_assignment
Assign assign(const std::vector<double>& cost, int rows, int cols, double thresh) {
    Assign r;
    std::vector<int> rc, cr;
    linear_assignment(cost, rows, cols, thresh, rc, cr);
    for (int i = 0; i < rows; ++i) {
        if (rc[i] >= 0)
            r.matches.emplace_back(i, rc[i]);
        else
            r.u_rows.push_back(i);
    }
    for (int j = 0; j < cols; ++j)
        if (cr[j] < 0) r.u_cols.push_back(j);
    return r;
}

// matching.iou_distance on boxes (float32), then optional fuse_score: 1 - IoU * score
std::vector<float> iou_cost(const std::vector<Box>& a, const Tracks& dets, bool fuse) {
    std::vector<float> c(a.size() * dets.size());
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < dets.size(); ++j) {
            const float d = 1.f - iou(a[i], dets[j]->xyxy());
            c[i * dets.size() + j] = fuse ? 1.f - (1.f - d) * dets[j]->score : d;
        }
    return c;
}

std::vector<Box> boxes_of(const Tracks& t) {
    std::vector<Box> b;
    b.reserve(t.size());
    for (const auto& x : t) b.push_back(x->xyxy());
    return b;
}

std::vector<double> to_double(const std::vector<float>& c) { return std::vector<double>(c.begin(), c.end()); }

// OCSORT._velocity_direction_cost
std::vector<float> direction_cost(const Tracks& tracks, const Tracks& dets) {
    std::vector<float> c(tracks.size() * dets.size(), 0.f);
    for (size_t i = 0; i < tracks.size(); ++i) {
        const auto& t = *tracks[i];
        if (!t.has_vel || t.last_obs[0] < 0.f) continue;
        const auto tc = center(t.last_obs);
        for (size_t j = 0; j < dets.size(); ++j) {
            const auto dc = center(dets[j]->xyxy());
            float dx = dc[0] - tc[0], dy = dc[1] - tc[1];
            const float n = std::sqrt(dx * dx + dy * dy);
            if (!(n > 1e-6f)) continue;
            dx /= n;
            dy /= n;
            float dot = dx * t.vel[0] + dy * t.vel[1];
            dot = std::min(1.f, std::max(-1.f, dot));
            c[i * dets.size() + j] = static_cast<float>(std::acos(dot) / kPi);
        }
    }
    return c;
}

// OCSORT.get_dists: IoU (+ score fuse) + inertia * direction cost
std::vector<double> get_dists(const Tracks& tracks, const Tracks& dets, const OCSortConfig& cfg) {
    auto c = iou_cost(boxes_of(tracks), dets, cfg.fuse_score);
    const auto vdc = direction_cost(tracks, dets);
    for (size_t k = 0; k < c.size(); ++k) c[k] = c[k] + cfg.inertia * vdc[k];
    return to_double(c);
}

Tracks joint(const Tracks& a, const Tracks& b) {  // stracks.joint_stracks
    Tracks r = a;
    std::unordered_set<int> ids;
    for (const auto& t : a) ids.insert(t->track_id);
    for (const auto& t : b)
        if (!ids.count(t->track_id)) r.push_back(t);
    return r;
}

Tracks sub(const Tracks& a, const Tracks& b) {  // stracks.sub_stracks
    std::unordered_set<int> ids;
    for (const auto& t : b) ids.insert(t->track_id);
    Tracks r;
    for (const auto& t : a)
        if (!ids.count(t->track_id)) r.push_back(t);
    return r;
}

}  // namespace

OCSort::OCSort(const OCSortConfig& cfg) : cfg_(cfg) {}
OCSort::~OCSort() = default;

void OCSort::reset() {
    tracked_.clear();
    lost_.clear();
    out_.clear();
    frame_id_ = 0;
    next_id_ = 0;
}

// BYTETracker.update with the OCSORT overrides
const std::vector<TrackOut>& OCSort::update(const std::vector<Det>& in) {
    ++frame_id_;
    const int fid = frame_id_;
    const int dt = cfg_.delta_t;
    Tracks activated, refind, lost_now;

    // _split_detections + init_track: detections become (not yet active) tracks
    Tracks dets;
    for (size_t i = 0; i < in.size(); ++i) {
        const Det& d = in[i];
        // Boxes.xywh in float32, then parse_bboxes promotes to float64 before xywh2ltwh, then float32
        const float xc = (d.x1 + d.x2) / 2.f, yc = (d.y1 + d.y2) / 2.f;
        const float w = d.x2 - d.x1, h = d.y2 - d.y1;
        if (!(w > 0.f && h > 0.f) || !(d.score >= cfg_.track_high_thresh)) continue;
        auto t = std::make_shared<Track>();
        t->tlwh0[0] = static_cast<float>(static_cast<double>(xc) - static_cast<double>(w) / 2.0);
        t->tlwh0[1] = static_cast<float>(static_cast<double>(yc) - static_cast<double>(h) / 2.0);
        t->tlwh0[2] = w;
        t->tlwh0[3] = h;
        t->score = d.score;
        t->cls = d.cls;
        t->idx = static_cast<int>(i);
        dets.push_back(std::move(t));
    }

    // _split_tracked: unconfirmed vs confirmed, then pool = confirmed + lost, all predicted
    Tracks unconfirmed, tracked;
    for (const auto& t : tracked_) (t->is_activated ? tracked : unconfirmed).push_back(t);
    Tracks pool = joint(tracked, lost_);
    for (auto& t : pool) {
        if (t->state != State::Tracked) t->mean[7] = 0.0;
        Kalman::predict(t->mean, t->cov);
    }

    // _first_association
    auto apply_match = [&](const TrackPtr& t, const TrackPtr& d) {  // BYTETracker._apply_match
        if (t->state == State::Tracked) {
            t->update(*d, fid, dt);
            activated.push_back(t);
        } else {
            t->re_activate(*d, fid, dt);
            refind.push_back(t);
        }
    };
    std::vector<int> u_track, u_det;
    {
        const auto c = get_dists(pool, dets, cfg_);
        Assign a = assign(c, static_cast<int>(pool.size()), static_cast<int>(dets.size()), cfg_.match_thresh);
        for (const auto& m : a.matches) apply_match(pool[m.first], dets[m.second]);
        u_track = a.u_rows;
        u_det = a.u_cols;
    }

    // _post_first_association: OCR on still-tracked unmatched tracks first, then on lost ones
    if (!u_det.empty()) {
        Tracks ocr_dets;
        for (int j : u_det) ocr_dets.push_back(dets[j]);
        std::vector<int> tr_idx, other_idx;
        for (int i : u_track) (pool[i]->state == State::Tracked ? tr_idx : other_idx).push_back(i);

        // OCSORT._ocr_associate; returns local unmatched (tracks, dets)
        auto ocr = [&](const std::vector<int>& tidx, const Tracks& ds, std::vector<int>& ut, std::vector<int>& ud) {
            ut.clear();
            ud.clear();
            if (tidx.empty() || ds.empty()) {
                for (size_t i = 0; i < tidx.size(); ++i) ut.push_back(static_cast<int>(i));
                for (size_t j = 0; j < ds.size(); ++j) ud.push_back(static_cast<int>(j));
                return;
            }
            std::vector<Box> tb;  // _ocr_distance: last real observation, else the predicted box
            for (int i : tidx) tb.push_back(pool[i]->last_obs[0] >= 0.f ? pool[i]->last_obs : pool[i]->xyxy());
            const auto c = to_double(iou_cost(tb, ds, cfg_.fuse_score));
            Assign a = assign(c, static_cast<int>(tidx.size()), static_cast<int>(ds.size()), cfg_.match_thresh);
            for (const auto& m : a.matches) {
                const TrackPtr& t = pool[tidx[m.first]];
                const TrackPtr& d = ds[m.second];
                if (t->state == State::Tracked) {
                    t->update(*d, fid, dt);
                    activated.push_back(t);
                } else {
                    t->apply_oru(d->xyxy(), fid);
                    t->re_activate(*d, fid, dt);
                    refind.push_back(t);
                }
            }
            ut = a.u_rows;
            ud = a.u_cols;
        };
        std::vector<int> ut1, ud1, ut2, ud2;
        ocr(tr_idx, ocr_dets, ut1, ud1);
        Tracks remaining;
        for (int j : ud1) remaining.push_back(ocr_dets[j]);
        ocr(other_idx, remaining, ut2, ud2);
        std::vector<int> nt, nd;
        for (int i : ut1) nt.push_back(tr_idx[i]);
        for (int i : ut2) nt.push_back(other_idx[i]);
        for (int j : ud2) nd.push_back(u_det[ud1[j]]);
        u_track = nt;
        u_det = nd;
    }

    // _second_association with use_byte=False: unmatched tracked tracks become lost
    for (int i : u_track) {
        const TrackPtr& t = pool[i];
        if (t->state == State::Tracked) {
            t->state = State::Lost;
            lost_now.push_back(t);
        }
    }

    // _unconfirmed_association (threshold 0.7)
    Tracks left;
    for (int j : u_det) left.push_back(dets[j]);
    std::vector<int> u_left;
    Tracks removed_now;
    if (unconfirmed.empty()) {
        for (size_t j = 0; j < left.size(); ++j) u_left.push_back(static_cast<int>(j));
    } else {
        const auto c = get_dists(unconfirmed, left, cfg_);
        Assign a = assign(c, static_cast<int>(unconfirmed.size()), static_cast<int>(left.size()), 0.7);
        for (const auto& m : a.matches) {
            unconfirmed[m.first]->update(*left[m.second], fid, dt);
            activated.push_back(unconfirmed[m.first]);
        }
        for (int i : a.u_rows) {
            unconfirmed[i]->state = State::Removed;
            removed_now.push_back(unconfirmed[i]);
        }
        u_left = a.u_cols;
    }

    // _init_new_tracks
    for (int j : u_left) {
        const TrackPtr& t = left[j];
        if (t->score < cfg_.new_track_thresh) continue;
        t->activate(fid, ++next_id_, dt);
        activated.push_back(t);
    }

    // _remove_stale_lost
    for (const auto& t : lost_)
        if (fid - t->frame_id > cfg_.track_buffer) t->state = State::Removed;

    // merge_track_pools
    Tracks tr;
    for (const auto& t : tracked_)
        if (t->state == State::Tracked) tr.push_back(t);
    tr = joint(tr, activated);
    tr = joint(tr, refind);
    Tracks lo = sub(lost_, tr);
    lo.insert(lo.end(), lost_now.begin(), lost_now.end());
    {
        Tracks keep;  // sub_stracks(lost, removed): removed tracks never come back
        for (const auto& t : lo)
            if (t->state != State::Removed) keep.push_back(t);
        lo.swap(keep);
    }
    // remove_duplicate_stracks(tracked, lost): IoU > 0.85 -> drop the shorter-lived one
    {
        std::vector<char> dup_a(tr.size(), 0), dup_b(lo.size(), 0);
        for (size_t p = 0; p < tr.size(); ++p) {
            const Box bp = tr[p]->xyxy();
            for (size_t q = 0; q < lo.size(); ++q) {
                if (1.f - iou(bp, lo[q]->xyxy()) < 0.15f) {
                    const int tp = tr[p]->frame_id - tr[p]->start_frame;
                    const int tq = lo[q]->frame_id - lo[q]->start_frame;
                    if (tp > tq)
                        dup_b[q] = 1;
                    else
                        dup_a[p] = 1;
                }
            }
        }
        Tracks ra, rb;
        for (size_t p = 0; p < tr.size(); ++p)
            if (!dup_a[p]) ra.push_back(tr[p]);
        for (size_t q = 0; q < lo.size(); ++q)
            if (!dup_b[q]) rb.push_back(lo[q]);
        tr.swap(ra);
        lo.swap(rb);
    }
    tracked_.swap(tr);
    lost_.swap(lo);

    // _format_output
    out_.clear();
    for (const auto& t : tracked_) {
        if (!t->is_activated) continue;
        const Box b = t->xyxy();
        out_.push_back({t->track_id, b[0], b[1], b[2], b[3], t->score, t->cls, t->idx});
    }
    return out_;
}

}  // namespace pt
