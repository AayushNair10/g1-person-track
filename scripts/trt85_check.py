"""TensorRT 8.5 compatibility check for ONNX files (shared by the export scripts).

    python3 scripts/trt85_check.py models/*.onnx

Checks: fixed shapes, opset <= 16, every op is one the TRT 8.5 ONNX parser imports, TopK with a constant K
and a float input, Mod (fmod=0) with integer inputs, no custom-domain ops.
"""
import collections
import sys
from pathlib import Path

import onnx

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


if __name__ == "__main__":
    bad = 0
    for p in sys.argv[1:]:
        r = inspect(Path(p))
        bad += bool(r["problems"])
        print("%-36s opset %d  %s" % (r["file"], r["opset"], "OK" if not r["problems"] else "; ".join(r["problems"])))
    sys.exit(1 if bad else 0)
