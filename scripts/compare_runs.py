#!/usr/bin/env python3
"""Compare person_track runs on the same recorded clip with a baseline run (Python 3.8 standard library only:
runs on the robot). Used by run_offline.sh to show what each compute saving costs in tracking.

    python3 scripts/compare_runs.py BASE.csv RUN.csv [RUN.csv ...] [--md OUT.md] [--json OUT.json]

Files deliver every frame, so frame k of two runs is the same picture. A run made with --every N (read from
its summary RUN.md.json) processed source frames 1, 1+N, ...; its frame k is the baseline's frame (k-1)*N+1.
Per run: frames where both / only one has a target, IoU of the two target boxes (>= 0.5 = same person, same
place), and how often the number of detected persons agrees.
"""
import argparse
import csv
import json
import os


def load(path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            box = None
            if r["target_id"] not in ("", "-1"):
                box = tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2"))
            rows[int(r["frame"])] = (box, int(r["n_persons"]))
    return rows


def every_of(csv_path):
    js = csv_path[:-len(".csv")] + ".md.json" if csv_path.endswith(".csv") else ""
    try:
        return int(json.load(open(js)).get("every", 1))
    except (OSError, ValueError):
        return 1


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def pct(v, q):
    if not v:
        return float("nan")
    v = sorted(v)
    pos = q * (len(v) - 1)
    i = int(pos)
    j = min(i + 1, len(v) - 1)
    return v[i] + (v[j] - v[i]) * (pos - i)


def compare(base, run, every):
    n = both = base_only = run_only = same_place = same_count = 0
    ious = []
    for k, (box, persons) in sorted(run.items()):
        ref = base.get((k - 1) * every + 1)
        if ref is None:
            continue
        n += 1
        same_count += ref[1] == persons
        if ref[0] and box:
            both += 1
            v = iou(ref[0], box)
            ious.append(v)
            same_place += v >= 0.5
        elif ref[0]:
            base_only += 1
        elif box:
            run_only += 1
    return {
        "frames": n, "both": both, "base_only": base_only, "run_only": run_only,
        "same_place_pct": 100.0 * same_place / both if both else float("nan"),
        "iou_mean": sum(ious) / len(ious) if ious else float("nan"), "iou_p5": pct(ious, 0.05),
        "same_count_pct": 100.0 * same_count / n if n else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args()
    base = load(a.base)
    out = {"base": os.path.basename(a.base), "runs": []}
    for p in a.runs:
        if os.path.abspath(p) == os.path.abspath(a.base):
            continue
        ev = every_of(p)
        r = compare(base, load(p), ev)
        r["run"] = os.path.basename(p)[:-len(".csv")]
        r["every"] = ev
        out["runs"].append(r)
    md = ["Baseline `%s`. Target = the box the app reports (same frames; `--every N` runs on their own frames)." %
          out["base"], "",
          "| run | frames | target in both | only baseline | only this run | same person & place % | "
          "target IoU mean / p5 | same person count % |",
          "|---|---|---|---|---|---|---|---|"]
    for r in out["runs"]:
        md.append("| %s | %d | %d | %d | %d | %.1f | %.3f / %.3f | %.1f |" % (
            r["run"], r["frames"], r["both"], r["base_only"], r["run_only"], r["same_place_pct"], r["iou_mean"],
            r["iou_p5"], r["same_count_pct"]))
    text = "\n".join(md) + "\n"
    print(text)
    if a.md:
        open(a.md, "w").write(text)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
