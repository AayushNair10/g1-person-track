// Clock and statistics helpers.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace pt {

// Seconds on the monotonic clock (the robot's wall clock may be unset, e.g. 1970).
inline double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Stat {
    double mean = 0, p50 = 0, p95 = 0, max = 0;
    size_t n = 0;
};

// Percentiles with linear interpolation (numpy's default).
inline Stat summarize(std::vector<double> v) {
    Stat s;
    s.n = v.size();
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    auto pct = [&](double q) {
        const double pos = q * static_cast<double>(v.size() - 1);
        const size_t i = static_cast<size_t>(std::floor(pos));
        const size_t j = std::min(i + 1, v.size() - 1);
        return v[i] + (v[j] - v[i]) * (pos - static_cast<double>(i));
    };
    double sum = 0;
    for (double x : v) sum += x;
    s.mean = sum / static_cast<double>(v.size());
    s.p50 = pct(0.50);
    s.p95 = pct(0.95);
    s.max = v.back();
    return s;
}

}  // namespace pt
