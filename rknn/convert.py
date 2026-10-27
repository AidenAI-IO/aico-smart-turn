"""Convert the Smart Turn v3 fp32 ONNX model to an int8 RKNN model.

Steps:
  1. Fix the dynamic batch dimension to 1 and simplify the graph with onnxsim,
     so the Shape/Slice/Concat subgraphs fold into constants.
  2. Quantize to int8 (w8a8) with rknn-toolkit2, calibrating on the features
     written by prepare_data.py, and export the .rknn file.
  3. Optionally run rknn-toolkit2's per-layer accuracy analysis on one sample.
  4. Optionally compare the RKNN model (run in the toolkit's simulator) with
     the fp32 ONNX model on the evaluation set written by prepare_data.py.

Start from the fp32 model (smart-turn-v3.2-gpu.onnx), not the int8 CPU model:
RKNN does its own quantization and cannot reuse onnxruntime's QDQ graph.
"""

import argparse
import json
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnxsim import simplify
from rknn.api import RKNN

INPUT_NAME = "input_features"


def fix_and_simplify(src, dst):
    """Fix the batch dimension to 1, simplify, and return (path, input shape).

    The mel-frame length (800 for an 8-second window) comes from the model.
    """
    model = onnx.load(src)
    for value in list(model.graph.input) + list(model.graph.output):
        dim0 = value.type.tensor_type.shape.dim[0]
        dim0.Clear()
        dim0.dim_value = 1
    input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    model, ok = simplify(model, overwrite_input_shapes={INPUT_NAME: input_shape})
    if not ok:
        raise RuntimeError("onnxsim could not validate the simplified model")
    onnx.save(model, dst)
    return dst, input_shape


def evaluate(rknn, onnx_path, eval_jsonl, limit=None):
    """Compare RKNN simulator output with fp32 ONNX on labelled samples."""
    with open(eval_jsonl) as f:
        rows = [json.loads(line) for line in f][:limit]

    session = ort.InferenceSession(onnx_path)
    if rknn.init_runtime() != 0:
        raise RuntimeError("init_runtime (simulator) failed")

    labels, p_onnx, p_rknn = [], [], []
    for row in rows:
        x = np.load(row["path"])
        labels.append(int(row["endpoint_bool"]))
        p_onnx.append(float(session.run(None, {INPUT_NAME: x})[0].reshape(-1)[0]))
        p_rknn.append(float(np.asarray(rknn.inference(inputs=[x])[0]).reshape(-1)[0]))

    labels, p_onnx, p_rknn = map(np.array, (labels, p_onnx, p_rknn))
    pred_onnx, pred_rknn = p_onnx > 0.5, p_rknn > 0.5
    summary = {
        "samples": len(rows),
        "accuracy_onnx_fp32": float((pred_onnx == labels).mean()),
        "accuracy_rknn": float((pred_rknn == labels).mean()),
        "decision_agreement": float((pred_onnx == pred_rknn).mean()),
        "prob_mean_abs_diff": float(np.abs(p_onnx - p_rknn).mean()),
        "prob_max_abs_diff": float(np.abs(p_onnx - p_rknn).max()),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", required=True, help="fp32 ONNX model (smart-turn-v3.2-gpu.onnx)")
    parser.add_argument("--calib", required=True, help="calib.txt written by prepare_data.py")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target", default="rv1106", help="rknn target_platform (rv1103 and rv1106 share an NPU)")
    parser.add_argument("--algorithm", default="mmse", choices=["normal", "mmse", "kl_divergence"],
                        help="rknn quantized_algorithm (mmse tracked fp32 most closely on the eval set)")
    parser.add_argument("--analyze", help="Optional .npy input for per-layer accuracy analysis")
    parser.add_argument("--eval", help="Optional eval.jsonl from prepare_data.py to compare against fp32 ONNX")
    parser.add_argument("--eval-limit", type=int, help="Evaluate only the first N samples")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # rknn-toolkit2 drops debug files (e.g. check3_fuse_ops.onnx) into the
    # working directory, so resolve inputs first and work inside out_dir.
    for name in ("onnx", "calib", "analyze", "eval"):
        if getattr(args, name):
            setattr(args, name, os.path.abspath(getattr(args, name)))
    os.chdir(args.out_dir)
    stem = os.path.splitext(os.path.basename(args.onnx))[0]
    fixed, input_shape = fix_and_simplify(args.onnx, f"{stem}-b1-sim.onnx")
    out_path = f"{stem}-{args.target}-int8-{args.algorithm}.rknn"

    rknn = RKNN(verbose=True)
    rknn.config(
        target_platform=args.target,
        quantized_dtype="w8a8",
        quantized_algorithm=args.algorithm,
        quantized_method="channel",
        optimization_level=3,
    )
    if rknn.load_onnx(model=fixed, inputs=[INPUT_NAME], input_size_list=[input_shape]) != 0:
        raise RuntimeError("load_onnx failed")
    if rknn.build(do_quantization=True, dataset=args.calib) != 0:
        raise RuntimeError("build failed")
    if rknn.export_rknn(out_path) != 0:
        raise RuntimeError("export_rknn failed")
    print(f"exported {os.path.abspath(out_path)}")

    if args.analyze:
        analysis_dir = f"accuracy-{args.algorithm}"
        rknn.accuracy_analysis(inputs=[args.analyze], output_dir=analysis_dir)
        print(f"accuracy analysis written to {os.path.abspath(analysis_dir)}")

    if args.eval:
        summary = evaluate(rknn, fixed, args.eval, args.eval_limit)
        print("EVAL " + json.dumps(summary))
        with open(os.path.splitext(out_path)[0] + ".eval.json", "w") as f:
            json.dump(summary, f, indent=2)

    rknn.release()


if __name__ == "__main__":
    main()
