#!/usr/bin/env python3
"""Compare and inspect live runs saved by scripts/run_live.sh (one folder per run: runs/<NNN>_<TAG>/).

    python3 scripts/analyze_runs.py [RUNS_DIR]          default: results/runs if it exists, else runs

Writes into RUNS_DIR:
  comparison.md / comparison.csv   one row per run: rate, latency, target stability, CPU / GPU / RAM / power
  comparison.png                   the same side by side                                   (needs matplotlib)
  <run>/analysis.png               one run over time: results per second, latency, target, load, power
The tables need only the Python standard library (works on the robot); the charts need matplotlib (the laptop
venv has it). Runs without a summary (stopped before the app wrote one) are skipped.
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_results import mean, pct, tegrastats_series  # noqa: E402

# chart colours: validated categorical slots 1-2 + neutral chrome (dataviz reference palette, light mode)
C1, C2 = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def read_run_txt(path):
    meta = {}
    if os.path.exists(path):
        for line in open(path, errors="replace"):
            k, sep, v = line.partition(": ")
            if sep:
                meta[k.strip()] = v.strip()
    return meta


def read_frames(path):
    cols = {"t": [], "target": [], "total_ms": [], "infer_ms": [], "persons": [], "event": []}
    if not os.path.exists(path):
        return cols
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                cols["t"].append(float(r["timestamp"]))
                cols["target"].append(int(r["target_id"]))
                cols["total_ms"].append(float(r["total_ms"]))
                cols["infer_ms"].append(float(r["infer_ms"]))
                cols["persons"].append(int(r["n_persons"]))
                cols["event"].append(r["event"])
            except (KeyError, ValueError):
                continue
    return cols


def load_runs(d):
    runs = []
    for name in sorted(os.listdir(d)):
        rd = os.path.join(d, name)
        js = os.path.join(rd, "summary.md.json")
        if not os.path.isdir(rd) or not os.path.exists(js):
            continue
        try:
            s = json.load(open(js))
        except ValueError:
            continue
        meta = read_run_txt(os.path.join(rd, "run.txt"))
        fr = read_frames(os.path.join(rd, "frames.csv"))
        ts_path = os.path.join(rd, "tegrastats.log")
        ts = tegrastats_series(ts_path) if os.path.exists(ts_path) else None
        runs.append({"name": name, "dir": rd, "s": s, "meta": meta, "frames": fr, "ts": ts})
    return runs


def variant(s):
    v = []
    eng = os.path.basename(s.get("engine", ""))
    if ".lean" in eng:
        v.append("lean")
    if s.get("dla", -1) >= 0:
        v.append("DLA%d" % s["dla"])
    if s.get("hw_resize"):
        v.append("hw-resize")
    if s.get("every", 1) > 1:
        v.append("every %d" % s["every"])
    return ", ".join(v) or "-"


def ts_mean(ts, key):
    if not ts:
        return float("nan")
    v = [x for x in ts[key] if x is not None]
    return mean(v) if v else float("nan")


def row_of(r):
    s, fr = r["s"], r["frames"]
    n = s.get("frames", 0)
    return {
        "run": r["name"],
        "note": r["meta"].get("note", ""),
        "engine": os.path.basename(s.get("engine", "")).replace(".engine", ""),
        "input": s.get("input", ""),
        "camera": s.get("camera_mode", s.get("frame_size", "")),
        "variant": variant(s),
        "seconds": (fr["t"][-1] - fr["t"][0]) if len(fr["t"]) > 1 else 0.0,
        "frames": n,
        "fps": s.get("fps", float("nan")),
        "camera_fps": s.get("camera_fps", float("nan")),
        "latency_p50": s["total"]["p50"], "latency_p95": s["total"]["p95"],
        "infer_p50": s["infer"]["p50"],
        "target_pct": 100.0 * s.get("with_target", 0) / n if n else float("nan"),
        "id_changes": s.get("id_changes", 0), "relocks": s.get("relocks", 0),
        "ids_seen": s.get("ids_seen", 0),
        "max_persons": max(fr["persons"]) if fr["persons"] else 0,
        "dropped": s.get("dropped", 0),
        "gpu_pct": ts_mean(r["ts"], "gpu"), "cpu_pct": ts_mean(r["ts"], "cpu"),
        "ram_mb": ts_mean(r["ts"], "ram"), "power_w": ts_mean(r["ts"], "power_w"),
        "rss_mb": s.get("rss_mb", float("nan")),
    }


def f(x, nd=1):
    return "-" if isinstance(x, float) and x != x else ("%.*f" % (nd, x) if isinstance(x, float) else str(x))


def write_tables(d, rows):
    cols = list(rows[0].keys())
    with open(os.path.join(d, "comparison.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    md = ["# Live runs compared", "",
          "Whole-robot CPU / GPU / RAM / power from tegrastats (Unitree's services included). Latency = camera frame "
          "read to result. Target ID changes = times the followed person got a new ID (re-locks: same place).", "",
          "| run | note | engine | camera | variant | s | results/s | camera FPS | latency p50 / p95 ms | "
          "infer p50 ms | target present % | target ID changes (re-locks) | persons max | GPU % | CPU % | "
          "RAM MB | power W | app RSS MB |",
          "|" + "---|" * 18]
    for r in rows:
        md.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s / %s | %s | %s | %d (%d) | %d | %s | %s | %s | %s | %s |" % (
            r["run"], r["note"] or "-", r["engine"], r["camera"], r["variant"], f(r["seconds"], 0), f(r["fps"]),
            f(r["camera_fps"]), f(r["latency_p50"]), f(r["latency_p95"]), f(r["infer_p50"], 2), f(r["target_pct"]),
            r["id_changes"], r["relocks"], r["max_persons"], f(r["gpu_pct"]), f(r["cpu_pct"]), f(r["ram_mb"], 0),
            f(r["power_w"], 2), f(r["rss_mb"], 0)))
    open(os.path.join(d, "comparison.md"), "w").write("\n".join(md) + "\n")
    return "\n".join(md)


# ---------------------------------------------------------------- charts (matplotlib, laptop)

def style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.edgecolor": AXIS,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.axisbelow": True, "legend.frameon": False,
    })


def run_chart(plt, r):
    fr, s = r["frames"], r["s"]
    if len(fr["t"]) < 2:
        return None
    t = fr["t"]
    span = t[-1]
    fig, ax = plt.subplots(5, 1, figsize=(10, 11), sharex=True, gridspec_kw={"hspace": 0.55})
    title = "%s   model input %s, camera %s" % (r["name"], s.get("input", ""), s.get("camera_mode", ""))
    if variant(s) != "-":
        title += ", " + variant(s)
    if r["meta"].get("note"):
        title += "\n" + r["meta"]["note"]
    fig.suptitle(title, x=0.06, ha="left", fontsize=11, fontweight="bold", color=INK)

    # 1. results per second (whole 1 s bins only)
    nb = max(1, int(span))
    counts = [0] * nb
    for x in t:
        if int(x) < nb:
            counts[int(x)] += 1
    a = ax[0]
    a.step([i + 1 for i in range(nb)], counts, where="pre", color=C1, lw=1.5)
    a.set_title("Results per second")
    a.set_ylim(0, max(counts) * 1.2 + 1)

    # 2. latency per frame, with p50 / p95
    a = ax[1]
    a.plot(t, fr["total_ms"], color=C1, lw=0.6)
    for q, lab in ((0.5, "p50"), (0.95, "p95")):
        v = pct(fr["total_ms"], q)
        a.axhline(v, color=INK2, lw=0.8, ls=(0, (4, 3)))
        a.text(span, v, " %s %.1f ms" % (lab, v), va="center", ha="left", color=INK2, fontsize=8)
    a.set_title("Latency, camera frame to result (ms)")
    top = pct(fr["total_ms"], 0.995) * 1.3
    a.set_ylim(0, max(top, 1.0))

    # 3. target present, with ID changes marked
    a = ax[2]
    pres = [1 if x >= 0 else 0 for x in fr["target"]]
    a.fill_between(t, pres, step="post", color=C1, alpha=0.35, lw=0)
    a.step(t, pres, where="post", color=C1, lw=1.0)
    prev = None
    for x, tid, ev in zip(t, fr["target"], fr["event"]):
        if tid >= 0 and tid != prev and ev in ("acquired", "relocked"):
            a.plot([x], [1.12], marker="v", ms=8, color=INK2, mec=SURFACE, mew=1.5, clip_on=False)
            a.text(x, 1.3, "id %d" % tid, ha="center", fontsize=7.5, color=INK2, clip_on=False)
        if tid >= 0:
            prev = tid
    a.set_ylim(0, 1.5)
    a.set_yticks([0, 1])
    a.set_yticklabels(["none", "present"])
    a.set_title("Target  (markers: new target ID)", pad=14)

    # 4 / 5. tegrastats, on the app's clock if both start times are known
    ts = r["ts"]
    off = 0.0
    try:
        off = float(r["meta"]["tegrastats_start_unix"]) - float(s["start_unix"])
    except (KeyError, ValueError):
        pass
    a = ax[3]
    a.set_title("Load, whole robot (%)")
    if ts and ts["gpu"]:
        tt = [off + i for i in range(len(ts["gpu"]))]
        ends = []
        for key, col, lab in (("gpu", C1, "GPU"), ("cpu", C2, "CPU (mean of cores)")):
            pts = [(x, y) for x, y in zip(tt, ts[key]) if y is not None and x <= span]
            if pts:
                a.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=1.5, label=lab)
                tail = [p[1] for p in pts[-10:]]
                ends.append([sum(tail) / len(tail), lab.split(" ")[0]])
        ends.sort()
        if len(ends) == 2 and ends[1][0] - ends[0][0] < 9:  # keep the end labels apart
            mid = (ends[0][0] + ends[1][0]) / 2
            ends[0][0], ends[1][0] = mid - 4.5, mid + 4.5
        for y, lab in ends:
            a.text(span, y, " " + lab, color=INK2, fontsize=8, va="center")
        a.legend(loc="upper left", ncol=2, fontsize=8)
        a.set_ylim(0, 100)
    a = ax[4]
    a.set_title("Board power, VDD_IN (W)")
    if ts and any(x is not None for x in ts["power_w"]):
        pts = [(off + i, y) for i, y in enumerate(ts["power_w"]) if y is not None]
        a.plot([p[0] for p in pts], [p[1] for p in pts], color=C1, lw=1.5)
        a.set_ylim(0, max(p[1] for p in pts) * 1.2)
    a.set_xlabel("seconds since the app started")
    a.set_xlim(0, span)
    out = os.path.join(r["dir"], "analysis.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def comparison_chart(plt, d, runs, rows):
    names = [r["run"] for r in rows][::-1]  # first run on top
    rows_r = rows[::-1]
    panels = [("Results per second", "fps", "%.1f"), ("Target present (% of frames)", "target_pct", "%.0f"),
              ("Target ID changes", "id_changes", "%d"), ("GPU load, mean (%)", "gpu_pct", "%.0f"),
              ("CPU load, mean of cores (%)", "cpu_pct", "%.0f"), ("Board power, mean (W)", "power_w", "%.1f"),
              ("RAM used, whole robot (MB)", "ram_mb", "%.0f")]
    h = max(2.2, 0.32 * len(names) + 1.2)
    fig, axs = plt.subplots(4, 2, figsize=(11, 4 * h), gridspec_kw={"hspace": 0.6, "wspace": 0.45})
    axs = axs.ravel()
    # latency distribution per run: box = p25-p75, line = p50, whiskers = p5 / p95
    a = axs[0]
    data = [r["frames"]["total_ms"] for r in runs][::-1]
    box = dict(whis=(5, 95), showfliers=False, widths=0.55, patch_artist=True,
               medianprops={"color": INK, "lw": 1.5}, whiskerprops={"color": C1, "lw": 1.2},
               capprops={"color": C1, "lw": 1.2}, boxprops={"facecolor": C1, "edgecolor": C1, "alpha": 0.55})
    try:
        a.boxplot(data, orientation="horizontal", **box)  # matplotlib >= 3.10
    except TypeError:
        a.boxplot(data, vert=False, **box)
    a.set_yticks(range(1, len(names) + 1))
    a.set_yticklabels(names)
    a.set_title("Latency per frame (ms): p5 / p25 / p50 / p75 / p95")
    a.grid(axis="y", visible=False)
    for a, (title, key, fm) in zip(axs[1:], panels):
        vals = [r[key] if r[key] == r[key] else 0 for r in rows_r]
        a.barh(range(len(names)), vals, height=0.62, color=C1)
        hi = max(vals) if vals and max(vals) > 0 else 1
        for i, (v, r) in enumerate(zip(vals, rows_r)):
            txt = "-" if r[key] != r[key] else fm % v
            a.text(v + hi * 0.02, i, txt, va="center", fontsize=8, color=INK)
        a.set_xlim(0, hi * 1.18)
        if key == "id_changes":
            a.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        a.set_yticks(range(len(names)))
        a.set_yticklabels(names)
        a.set_title(title)
        a.grid(axis="y", visible=False)
    fig.suptitle("Live runs compared (details: comparison.md)", x=0.06, ha="left", fontsize=12, fontweight="bold")
    out = os.path.join(d, "comparison.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_dir", nargs="?")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    d = a.runs_dir or next((p for p in (os.path.join(root, "results", "runs"), os.path.join(root, "runs"))
                            if os.path.isdir(p)), None)
    if not d or not os.path.isdir(d):
        sys.exit("no runs folder (give its path)")
    runs = load_runs(d)
    if not runs:
        sys.exit("no finished runs in " + d)
    rows = [row_of(r) for r in runs]
    print(write_tables(d, rows))
    print("\nwrote %s and %s" % (os.path.join(d, "comparison.md"), os.path.join(d, "comparison.csv")))
    if a.no_plots:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available: tables only (run this on the laptop for the charts)")
        return
    style(plt)
    for r in runs:
        out = run_chart(plt, r)
        if out:
            print("wrote " + out)
    print("wrote " + comparison_chart(plt, d, runs, rows))


if __name__ == "__main__":
    main()
