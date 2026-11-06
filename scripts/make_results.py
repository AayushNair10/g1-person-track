#!/usr/bin/env python3
"""Collect the robot's logs into logs/RESULTS.md (Python 3.8 standard library only: runs on the robot).

    python3 scripts/make_results.py [logs_dir]

Reads: logs/phase1.txt, logs/trtexec_*.log, logs/*.md.json (person_track summaries), logs/tegrastats_*.log,
logs/agreement.json (scripts/compare_runs.py)
"""
import glob
import json
import os
import re
import sys


def pct(v, q):
    if not v:
        return float("nan")
    v = sorted(v)
    pos = q * (len(v) - 1)
    i = int(pos)
    j = min(i + 1, len(v) - 1)
    return v[i] + (v[j] - v[i]) * (pos - i)


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def trtexec_rows(d):
    rows = []
    for log in sorted(glob.glob(os.path.join(d, "trtexec_*.log"))):
        name = os.path.basename(log)[len("trtexec_"):-len(".log")]
        txt = open(log, errors="replace").read()
        m = re.findall(r"GPU Compute Time: (.*)", txt)
        q = re.findall(r"Throughput: ([0-9.]+) qps", txt)
        if not m:
            err = re.findall(r"(?i)(error[^\n]*)", txt)
            rows.append((name, "failed", "", "", "", (err[-1][:90] if err else "")))
            continue
        line = m[-1]

        def get(key):
            r = re.search(re.escape(key) + r" = ([0-9.]+)", line)
            return r.group(1) if r else ""

        rows.append((name, get("median"), get("mean"), get("percentile(99%)"), q[-1] if q else "", ""))
    return rows


def tegrastats(path):
    gpu, ram, cpu, power, emc = [], [], [], [], []
    for line in open(path, errors="replace"):
        m = re.search(r"GR3D_FREQ (\d+)%", line)
        if m:
            gpu.append(int(m.group(1)))
        m = re.search(r"RAM (\d+)/(\d+)MB", line)
        if m:
            ram.append((int(m.group(1)), int(m.group(2))))
        m = re.search(r"CPU \[([^\]]+)\]", line)
        if m:
            loads = [int(x) for x in re.findall(r"(\d+)%", m.group(1))]
            if loads:
                cpu.append(mean(loads))
        m = re.search(r"VDD_IN (\d+)mW", line)
        if m:
            power.append(int(m.group(1)))
        m = re.search(r"EMC_FREQ (\d+)%", line)
        if m:
            emc.append(int(m.group(1)))
    return {
        "samples": max(len(gpu), len(ram)),
        "gpu_mean": mean(gpu), "gpu_p95": pct(gpu, 0.95), "gpu_max": max(gpu) if gpu else float("nan"),
        "ram_mean": mean([r[0] for r in ram]), "ram_max": max([r[0] for r in ram]) if ram else float("nan"),
        "ram_total": ram[0][1] if ram else float("nan"),
        "cpu_mean": mean(cpu), "power_mean_w": mean(power) / 1000 if power else float("nan"),
        "emc_mean": mean(emc),
    }


def fmt(x, nd=2):
    if isinstance(x, (int, float)):
        return "-" if x != x else ("%.*f" % (nd, x))
    return str(x)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
    d = os.path.abspath(d)
    out = ["# G1 person tracking: results", ""]

    p1 = os.path.join(d, "phase1.txt")
    if os.path.exists(p1):
        out += ["## Phase 1: robot check", "", "```", open(p1, errors="replace").read().strip(), "```", ""]

    rows = trtexec_rows(d)
    if rows:
        out += ["## Phase 3: TensorRT engines (trtexec, GPU compute time, Unitree services running)", "",
                "| engine | median ms | mean ms | p99 ms | throughput qps | note |", "|---|---|---|---|---|---|"]
        out += ["| %s | %s | %s | %s | %s | %s |" % r for r in rows]
        out.append("")

    runs = []
    for js in sorted(glob.glob(os.path.join(d, "*.md.json"))):
        try:
            r = json.load(open(js))
            r["_file"] = os.path.basename(js)[:-len(".md.json")]
            runs.append(r)
        except ValueError:
            pass

    comp = [r for r in runs if "compare" in r]
    if comp:
        out += ["## Phase 5a: fp16 vs fp32 on the same frames", "",
                "| run | frames | detections fp16 / fp32 | matched | mean IoU | min IoU | max score diff | same count % |",
                "|---|---|---|---|---|---|---|---|"]
        for r in comp:
            c = r["compare"]
            out.append("| %s | %d | %d / %d | %d | %.4f | %.4f | %.4f | %.1f |" % (
                r["label"], c["frames"], c["n_a"], c["n_b"], c["matched"], c["iou_mean"], c["iou_min"],
                c["dconf_max"], c["same_count_pct"]))
        out.append("")

    def variant(r):
        v = []
        if ".lean" in r.get("engine", ""):
            v.append("lean")
        if r.get("dla", -1) >= 0:
            v.append("DLA%d" % r["dla"])
        if r.get("hw_resize"):
            v.append("hw-resize " + r.get("frame_size", ""))
        if r.get("every", 1) > 1:
            v.append("every %d" % r["every"])
        return ", ".join(v) or "-"

    def run_table(title, rs):
        if not rs:
            return []
        t = ["## " + title, "",
             "| run | input | variant | head | FPS | frames | latency p50 / p95 ms | infer p50 / p95 ms | pre p50 ms | "
             "post p50 ms | track p50 ms | target ID changes | frames without target | dropped |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rs:
            t.append("| %s | %s | %s | %s | %.2f | %d | %.2f / %.2f | %.3f / %.3f | %.3f | %.3f | %.3f | %d (re-locks %d) | %d | %d |" % (
                r["label"], r["input"], variant(r), r["head"], r["fps"], r["frames"], r["total"]["p50"], r["total"]["p95"],
                r["infer"]["p50"], r["infer"]["p95"], r["pre"]["p50"], r["post"]["p50"], r["track"]["p50"],
                r["id_changes"], r["relocks"], r["without_target"], r["dropped"]))
        return t + [""]

    out += run_table("Phase 5a: offline throughput (recorded clip, every frame, no camera limit)",
                     [r for r in runs if r["_file"].startswith("offline_")])
    ag = os.path.join(d, "agreement.json")
    if os.path.exists(ag):
        try:
            a = json.load(open(ag))
            out += ["## Phase 5a: what each saving changes (target vs the baseline `%s`, same clip)" % a["base"], "",
                    "| run | frames | target in both | only baseline | only this run | same person & place % | "
                    "target IoU mean / p5 | same person count % |", "|---|---|---|---|---|---|---|---|"]
            for r in a["runs"]:
                out.append("| %s | %d | %d | %d | %d | %s | %s / %s | %s |" % (
                    r["run"], r["frames"], r["both"], r["base_only"], r["run_only"], fmt(r["same_place_pct"], 1),
                    fmt(r["iou_mean"], 3), fmt(r["iou_p5"], 3), fmt(r["same_count_pct"], 1)))
            out.append("")
        except (ValueError, KeyError):
            pass
    live = [r for r in runs if r["_file"].startswith("live_")]
    out += run_table("Phase 5b: live head camera (camera limits to ~15 FPS)", live)

    ts_rows = []
    for log in sorted(glob.glob(os.path.join(d, "tegrastats_*.log"))):
        s = tegrastats(log)
        if s["samples"]:
            ts_rows.append((os.path.basename(log)[len("tegrastats_"):-len(".log")], s))
    if ts_rows:
        out += ["### tegrastats during the live runs (whole robot, incl. Unitree services)", "",
                "| run | samples | GPU % mean / p95 / max | RAM used MB mean / max (of total) | CPU % mean | EMC % | VDD_IN W |",
                "|---|---|---|---|---|---|---|"]
        for tag, s in ts_rows:
            out.append("| %s | %d | %s / %s / %s | %s / %s (%s) | %s | %s | %s |" % (
                tag, s["samples"], fmt(s["gpu_mean"], 1), fmt(s["gpu_p95"], 0), fmt(s["gpu_max"], 0),
                fmt(s["ram_mean"], 0), fmt(s["ram_max"], 0), fmt(s["ram_total"], 0), fmt(s["cpu_mean"], 1),
                fmt(s["emc_mean"], 1), fmt(s["power_mean_w"], 2)))
        out.append("")

    md_runs = sorted(glob.glob(os.path.join(d, "*.md")))
    md_runs = [m for m in md_runs if os.path.basename(m) not in ("RESULTS.md", "engines.md", "agreement.md")]
    if md_runs:
        out += ["## Full run summaries", ""]
        for m in md_runs:
            out += [open(m, errors="replace").read().strip().replace("## person_track run", "### run"), ""]

    path = os.path.join(d, "RESULTS.md")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote " + path)


if __name__ == "__main__":
    main()
