// Desktop check of the C++ OC-SORT + target selector (no OpenCV, no GPU).
//   test_tracker <dets.txt> <tracks_out.txt>
// dets.txt (from check_tracker.py):  "F <frame> <n>" then n lines "x1 y1 x2 y2 score cls"
// tracks_out.txt:                    "F <frame> <n> <target_id>" then n lines "id x1 y1 x2 y2 score det_index"
// Prints a target summary (ID changes, frames without a target) on stdout.
#include <cstdio>
#include <vector>

#include "../../src/ocsort.hpp"
#include "../../src/target.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s dets.txt tracks_out.txt\n", argv[0]);
        return 2;
    }
    std::FILE* in = std::fopen(argv[1], "r");
    std::FILE* out = std::fopen(argv[2], "w");
    if (!in || !out) {
        std::fprintf(stderr, "cannot open files\n");
        return 2;
    }
    pt::OCSort tracker;
    pt::TargetSelector target;
    std::vector<pt::Det> dets;
    int frame = 0, n = 0, frames = 0, no_target = 0;
    while (std::fscanf(in, " F %d %d", &frame, &n) == 2) {
        dets.resize(static_cast<size_t>(n));
        for (auto& d : dets)
            if (std::fscanf(in, "%f %f %f %f %f %d", &d.x1, &d.y1, &d.x2, &d.y2, &d.score, &d.cls) != 6) {
                std::fprintf(stderr, "bad detection line in frame %d\n", frame);
                return 2;
            }
        const auto& tracks = tracker.update(dets);
        pt::TrackOut t{};
        const bool has = target.update(tracks, t);
        ++frames;
        no_target += !has;
        std::fprintf(out, "F %d %zu %d\n", frame, tracks.size(), has ? t.id : -1);
        for (const auto& r : tracks)
            std::fprintf(out, "%d %.9g %.9g %.9g %.9g %.9g %d\n", r.id, r.x1, r.y1, r.x2, r.y2, r.score, r.det_index);
    }
    std::fclose(in);
    std::fclose(out);
    std::printf("frames %d  no_target %d  target_id_changes %d (relocks %d, reacquires %d)  last_target_id %d\n",
                frames, no_target, target.id_changes(), target.relocks(), target.reacquires(), target.target_id());
    return 0;
}
