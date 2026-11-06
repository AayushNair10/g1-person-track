"""Phase 2 (laptop): export YOLO .pt files to fixed-size ONNX for TensorRT 8.5 and inspect them.

    ~/Downloads/benchmark_face/.venv/bin/python scripts/export_models.py \
        --weights ~/Downloads/benchmark_face --out models

For each (model, size) this runs the equivalent of
    yolo export model=<m>.pt format=onnx opset=16 imgsz=<h>,<w> dynamic=False simplify=True
(plus nms=False for YOLO26's NMS-free head), renames the file to models/<name>_<w>x<h>.onnx and checks
it for TensorRT 8.5: fixed shapes, opset <= 16, every op is one the TRT 8.5 ONNX parser imports (no
LayerNormalization), TopK has a constant K and a float input, Mod (fmod=0) has integer inputs.
A JSON report with names, shapes and op counts goes to models/onnx_report.json.

Outputs: yolo26n_*      NMS-free head, (1, 300, 6) = x1 y1 x2 y2 score class (input pixels), no NMS
         yolo26n-raw_*  fallback 1: same weights, raw one-to-many head (1, 84, N), NMS in C++
         yolo11n_*      fallback 2: raw head (1, 84, N), NMS in C++

Sizes: the square 640/416/320 asked for, plus 16:9-friendly rectangles (WxH 640x384, 416x256, 320x192).
The head-camera stream is 1280x720, so a 640x640 input is 44% grey padding; 640x384 holds the same
640x360 picture with 24 px of padding and needs ~40% less compute.
"""
import argparse
import collections
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "False")  # never pip-install anything behind our back
os.environ.setdefault("YOLO_OFFLINE", "1")

import onnx  # noqa: E402
from onnx import numpy_helper  # noqa: E402

# Ops the TensorRT 8.5 ONNX parser can import: every DEFINE_BUILTIN_OP_IMPORTER in onnx-tensorrt
# release/8.5-GA builtin_op_importers.cpp (the parser shipped with TRT 8.5.2). Anything else ends in
# importFallbackPluginImporter and fails. LayerNormalization is absent here (it arrived in TRT 8.6).
TRT85_SUPPORTED = set(
    "Abs Acos Acosh Add And ArgMax ArgMin Asin Asinh Atan Atanh AveragePool BatchNormalization Cast Ceil Celu "
    "Clip Concat Constant ConstantOfShape Conv ConvTranspose Cos Cosh CumSum DepthToSpace DequantizeLinear Div "
    "Dropout Einsum Elu Equal Erf Exp Expand EyeLike Flatten Floor GRU Gather GatherElements GatherND Gemm "
    "GlobalAveragePool GlobalLpPool GlobalMaxPool Greater GreaterOrEqual GridSample HardSigmoid HardSwish "
    "Identity If ImageScaler InstanceNormalization IsNaN LRN LSTM LeakyRelu Less LessOrEqual Log LogSoftmax "
    "Loop LpNormalization LpPool MatMul Max MaxPool Mean MeanVarianceNormalization Min Mod Mul Neg "
    "NonMaxSuppression NonZero Not OneHot Or PRelu Pad ParametricSoftplus Pow QuantizeLinear RNN RandomNormal "
    "RandomNormalLike RandomUniform RandomUniformLike Range Reciprocal ReduceL1 ReduceL2 ReduceLogSum "
    "ReduceLogSumExp ReduceMax ReduceMean ReduceMin ReduceProd ReduceSum ReduceSumSquare Relu Reshape Resize "
    "ReverseSequence RoiAlign Round ScaledTanh Scan Scatter ScatterElements ScatterND Selu Shape Shrink Sigmoid "
    "Sign Sin Sinh Size Slice Softmax Softplus Softsign SpaceToDepth Split Sqrt Squeeze Sub Sum Tan Tanh "
    "ThresholdedRelu Tile TopK Transpose Trilu Unsqueeze Upsample Where Xor".split()
)
INT_TYPES = {onnx.TensorProto.INT32, onnx.TensorProto.INT64, onnx.TensorProto.INT8, onnx.TensorProto.UINT8}

# (file name, weights, (height, width), nms-free head?) to export
DEFAULT_JOBS = [
    ("yolo26n", "yolo26n", (640, 640), True), ("yolo26n", "yolo26n", (416, 416), True),
    ("yolo26n", "yolo26n", (320, 320), True), ("yolo26n", "yolo26n", (384, 640), True),
    ("yolo26n", "yolo26n", (256, 416), True), ("yolo26n", "yolo26n", (192, 320), True),
    ("yolo26n-raw", "yolo26n", (640, 640), False), ("yolo26n-raw", "yolo26n", (384, 640), False),
    ("yolo11n", "yolo11n", (640, 640), False), ("yolo11n", "yolo11n", (384, 640), False),
]


def dims(vi):
    return [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in vi.type.tensor_type.shape.dim]


def elem_types(m):
    """Element type of every tensor (after shape inference)."""
    m = onnx.shape_inference.infer_shapes(m)
    t = {}
    for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        t[vi.name] = vi.type.tensor_type.elem_type
    for i in m.graph.initializer:
        t[i.name] = i.data_type
    for n in m.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    t[n.output[0]] = a.t.data_type
    return t


def inspect(path):
    """Return a report dict and a list of problems for TensorRT 8.5."""
    m = onnx.load(str(path))
    g = m.graph
    inits = {i.name for i in g.initializer}
    consts = {n.output[0] for n in g.node if n.op_type == "Constant"}
    opset = max((o.version for o in m.opset_import if o.domain in ("", "ai.onnx")), default=0)
    ops = collections.Counter(n.op_type for n in g.node)
    problems = []
    if opset > 16:
        problems.append(f"opset {opset} > 16")
    for op in sorted(ops):
        if op not in TRT85_SUPPORTED:
            problems.append(f"op {op} x{ops[op]} not in the TRT 8.5 ONNX parser")
    types = elem_types(m)
    for n in g.node:
        if n.op_type == "TopK":
            if len(n.input) > 1 and n.input[1] not in inits | consts:
                problems.append(f"TopK {n.name}: K is not a constant")
            if types.get(n.input[0]) in INT_TYPES:
                problems.append(f"TopK {n.name}: integer input (TRT 8.5 TopK needs float)")
        if n.op_type == "Mod":
            fmod = next((a.i for a in n.attribute if a.name == "fmod"), 0)
            if fmod == 0 and types.get(n.input[0]) not in INT_TYPES:
                problems.append(f"Mod {n.name}: fmod=0 on non-integer input {types.get(n.input[0])}")
        if n.domain not in ("", "ai.onnx"):
            problems.append(f"custom-domain op {n.domain}:{n.op_type}")
    ins = [(i.name, dims(i)) for i in g.input if i.name not in inits]
    outs = [(o.name, dims(o)) for o in g.output]
    for name, d in ins + outs:
        if any(not isinstance(v, int) or v <= 0 for v in d):
            problems.append(f"{name} has a dynamic shape {d}")
    int64_inits = sum(1 for i in g.initializer if i.data_type == onnx.TensorProto.INT64)
    meta = {p.key: p.value for p in m.metadata_props}
    rep = {
        "file": path.name,
        "size_mb": round(path.stat().st_size / 1e6, 2),
        "opset": opset,
        "inputs": ins,
        "outputs": outs,
        "n_nodes": len(g.node),
        "ops": dict(sorted(ops.items())),
        "int64_initializers": int64_inits,  # TRT casts these to INT32 (harmless warning)
        "end2end": meta.get("end2end"),
        "problems": problems,
    }
    try:
        onnx.checker.check_model(m)
    except Exception as e:  # pragma: no cover
        problems.append(f"onnx.checker: {e}")
    return rep


def export(weights_dir, out_dir, name, model, hw, nms_free, opset):
    from ultralytics import YOLO

    src = Path(weights_dir) / f"{model}.pt"
    pt = out_dir / f"{model}.pt"
    if not pt.exists():
        shutil.copy2(src, pt)
    h, w = hw
    dst = out_dir / f"{name}_{w}x{h}.onnx"
    # nms=False selects YOLO26's NMS-free (one-to-one) head, the one `.pt` predict uses and the one the
    # Mac benchmark ran. (Ultralytics 8.4 defaults to nms=None = raw one-to-many head that needs NMS.)
    # YOLO11/YOLOv8 have no such head; they always export raw (1, 84, N) and need NMS in C++.
    f = YOLO(str(pt)).export(format="onnx", opset=opset, imgsz=[h, w], dynamic=False, simplify=True,
                             nms=False if nms_free else None, batch=1, device="cpu", verbose=False)
    shutil.move(f, dst)
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(Path.home() / "Downloads/benchmark_face"), help="folder with <model>.pt")
    ap.add_argument("--out", default="models")
    ap.add_argument("--opset", type=int, default=16)
    ap.add_argument("--only", default="", help="comma list of file stems to (re)export, e.g. yolo26n_640x384")
    ap.add_argument("--inspect-only", action="store_true", help="just inspect existing models/*.onnx")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.inspect_only:
        only = {s for s in args.only.split(",") if s}
        for name, model, hw, nms_free in DEFAULT_JOBS:
            stem = f"{name}_{hw[1]}x{hw[0]}"
            if only and stem not in only:
                continue
            print(f"[export] {stem} ...", flush=True)
            export(args.weights, out, name, model, hw, nms_free, args.opset)

    reports, bad = [], 0
    for p in sorted(out.glob("*.onnx")):
        r = inspect(p)
        reports.append(r)
        status = "OK" if not r["problems"] else "PROBLEMS: " + "; ".join(r["problems"])
        bad += bool(r["problems"])
        print(f"\n== {p.name}  ({r['size_mb']} MB, opset {r['opset']}, {r['n_nodes']} nodes, end2end={r['end2end']})")
        print(f"   inputs  {r['inputs']}\n   outputs {r['outputs']}")
        print(f"   ops     {r['ops']}")
        print(f"   TRT 8.5 {status}")
    (out / "onnx_report.json").write_text(json.dumps(reports, indent=1))
    print(f"\nreport: {out / 'onnx_report.json'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
