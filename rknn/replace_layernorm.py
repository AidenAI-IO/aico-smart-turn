"""Replace Smart Turn's encoder LayerNorms with an NPU-friendly normalization.

On the RV1106 the runtime executes LayerNorm on the CPU (~26 ms per layer),
and forcing it onto the NPU produces wrong results. This script swaps the
encoder's nine LayerNorms for Dynamic Tanh (DyT, "Transformers without
Normalization", Zhu et al. 2025):

    DyT(x) = weight * tanh(alpha * (x - shift)) + bias

DyT is purely element-wise, so it compiles to NPU Mul/Add/Tanh ops. The
classifier head's LayerNorm (one 256-value vector, ~0.1 ms on the CPU) is kept.

The student starts from the released v3.2 weights (recovered from the fp32
ONNX file, since only ONNX weights are published) and is fine-tuned by
distilling the original model's probabilities and per-layer residual streams,
plus the ground-truth labels.

With --seconds 4 the student also gets a shorter input window (200 encoder
positions instead of 400), which cuts attention cost about 4x on the NPU. The
teacher still sees the full 8 seconds.

Subcommands:
  features  compute training features from smart-turn-data-v3.2-train shards
  train     calibrate DyT, fine-tune, export ONNX, evaluate on eval.jsonl
  evaluate  evaluate an exported ONNX model on eval.jsonl
"""

import argparse
import io
import json
import math
import os
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import pyarrow.parquet as pq
import soundfile as sf
import torch
from onnx import numpy_helper
from torch import nn
from transformers import WhisperConfig, WhisperFeatureExtractor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from prepare_data import SAMPLE_RATE, compute_features  # noqa: E402
from train import CONFIG, SmartTurnV3Model, export_to_onnx_fp32  # noqa: E402

N_LAYERS = 4
POSITIONS_PER_SECOND = 50  # 100 mel frames per second, halved by conv2's stride


def model_name(seconds):
    return "smart-turn-v3.2-dyt" + ("" if seconds == 8 else f"-{seconds}s")


# --------------------------------------------------------------------------
# Reference model

def load_reference_model(onnx_path):
    """Rebuild SmartTurnV3Model with the weights stored in the fp32 ONNX export."""
    config = WhisperConfig.from_pretrained(CONFIG["base_model_name"])
    model = SmartTurnV3Model(config)

    graph = onnx.load(onnx_path).graph
    weights = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    state = {name[len("inner."):]: w for name, w in weights.items() if name.startswith("inner.")}

    # The exporter stored nn.Linear weights of the encoder and the pooling head
    # as transposed, anonymous MatMul constants (val_N), in module order:
    # q, k, v, out, fc1, fc2 for each layer, then pool_attention.0 / .2.
    anonymous = sorted((n for n in weights if n.startswith("val_") and weights[n].ndim == 2),
                       key=lambda n: int(n[4:]))
    linear_names = [f"encoder.layers.{i}.{p}" for i in range(N_LAYERS)
                    for p in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                              "self_attn.out_proj", "fc1", "fc2")]
    linear_names += ["pool_attention.0", "pool_attention.2"]
    if len(anonymous) != len(linear_names):
        raise ValueError(f"expected {len(linear_names)} MatMul weights, found {len(anonymous)}")
    for name, key in zip(linear_names, anonymous):
        state[f"{name}.weight"] = weights[key].T

    tensors = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in state.items()}
    missing, unexpected = model.load_state_dict(tensors, strict=False)
    # k_proj has no bias in Whisper; nothing else may be missing.
    if unexpected or any("k_proj.bias" not in m for m in missing):
        raise ValueError(f"weight mapping mismatch: missing={missing} unexpected={unexpected}")
    return model.eval()


# --------------------------------------------------------------------------
# Dynamic Tanh

class DynamicTanh(nn.Module):
    """Element-wise replacement for LayerNorm: weight * tanh(alpha * (x - shift)) + bias."""

    def __init__(self, layer_norm: nn.LayerNorm):
        super().__init__()
        dim = layer_norm.normalized_shape[0]
        device = layer_norm.weight.device
        self.alpha = nn.Parameter(torch.ones(dim, device=device))
        self.shift = nn.Parameter(torch.zeros(dim, device=device))
        self.weight = nn.Parameter(layer_norm.weight.detach().clone())
        self.bias = nn.Parameter(layer_norm.bias.detach().clone())

    def forward(self, x):
        return self.weight * torch.tanh(self.alpha * (x - self.shift)) + self.bias


def encoder_layer_norms(model):
    """(parent module, attribute name) of every LayerNorm inside the encoder."""
    found = []
    for layer in model.encoder.layers:
        found += [(layer, "self_attn_layer_norm"), (layer, "final_layer_norm")]
    found.append((model.encoder, "layer_norm"))
    return found


@torch.no_grad()
def replace_with_dyt(model, calib_batch, gain):
    """Swap encoder LayerNorms for DyT, initialised to mimic each LayerNorm.

    For each LayerNorm, record its input statistics on calib_batch. DyT then
    starts as LN with *fixed* statistics: shift = mean of the token means,
    and alpha = 1 / (gain * typical token std) with weight scaled by gain, so
    tanh stays near its linear range for typical activations.
    """
    stats = {}
    hooks = []
    for parent, attr in encoder_layer_norms(model):
        ln = getattr(parent, attr)

        def hook(module, inputs, _out, key=id(ln)):
            x = inputs[0].float()
            stats[key] = (x.mean(-1).mean().item(), x.std(-1).median().item())

        hooks.append(ln.register_forward_hook(hook))
    model(calib_batch)
    for h in hooks:
        h.remove()

    for parent, attr in encoder_layer_norms(model):
        ln = getattr(parent, attr)
        mean, std = stats[id(ln)]
        dyt = DynamicTanh(ln)
        dyt.shift.fill_(mean)
        dyt.alpha.fill_(1.0 / (gain * std))
        dyt.weight.mul_(gain)
        setattr(parent, attr, dyt)
    return model


def shorten_window(model, seconds):
    """Make the encoder accept only the last `seconds` of the 8-second window.

    Inputs are aligned to the end of the audio (padding goes at the front), so
    the kept positional embeddings are the *last* ones: each position keeps its
    distance from the end of the user's speech.
    """
    encoder = model.encoder
    n = seconds * POSITIONS_PER_SECOND
    if n == encoder.embed_positions.num_embeddings:
        return model
    old = encoder.embed_positions.weight.detach()
    encoder.embed_positions = nn.Embedding(n, old.shape[1], device=old.device)
    encoder.embed_positions.weight.data.copy_(old[-n:])
    encoder.embed_positions.requires_grad_(False)  # frozen, as in WhisperEncoder
    encoder.config.max_source_positions = n
    encoder.max_source_positions = n
    return model


# --------------------------------------------------------------------------
# Data

def cmd_features(args):
    """Decode parquet shards and store (N, 80, 100 * seconds) float16 features + labels."""
    extractor = WhisperFeatureExtractor(chunk_length=args.seconds)
    labels, total = [], 0
    for path in args.parquet:
        total += pq.ParquetFile(path).metadata.num_rows
    feats = np.lib.format.open_memmap(args.out + ".features.npy", mode="w+",
                                      dtype=np.float16, shape=(total, 80, 100 * args.seconds))
    n = 0
    for path in args.parquet:
        table = pq.read_table(path, columns=["audio", "endpoint_bool"])
        audio, endpoint = table.column("audio"), table.column("endpoint_bool")
        for i in range(table.num_rows):
            data, sr = sf.read(io.BytesIO(audio[i]["bytes"].as_py()), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != SAMPLE_RATE:
                continue
            feats[n] = compute_features(extractor, data, args.seconds)[0]
            labels.append(int(endpoint[i].as_py()))
            n += 1
        print(f"{path}: {n} samples so far", flush=True)
    feats.flush()
    np.save(args.out + ".labels.npy", np.array(labels, dtype=np.int8))
    with open(args.out + ".count", "w") as f:
        f.write(str(n))
    print(f"wrote {n} samples (positives: {sum(labels)})")


def load_features(prefix):
    with open(prefix + ".count") as f:
        n = int(f.read())
    feats = np.load(prefix + ".features.npy", mmap_mode="r")[:n]
    labels = np.load(prefix + ".labels.npy")[:n]
    return feats, labels


def load_eval(eval_jsonl):
    rows = [json.loads(line) for line in open(eval_jsonl)]

    def resolve(path):
        # prepare_data.py records absolute paths, which differ inside and
        # outside its container; fall back to <jsonl dir>/eval/<file>.
        if os.path.exists(path):
            return path
        return os.path.join(os.path.dirname(os.path.abspath(eval_jsonl)),
                            os.path.basename(os.path.dirname(path)), os.path.basename(path))

    x = np.concatenate([np.load(resolve(r["path"])) for r in rows]).astype(np.float32)
    y = np.array([int(r["endpoint_bool"]) for r in rows])
    return x, y


# --------------------------------------------------------------------------
# Training

def to_logit(prob):
    """SmartTurnV3Model returns sigmoid probabilities; recover logits for distillation."""
    return torch.logit(prob.float().clamp(1e-6, 1 - 1e-6))


class LayerOutputRecorder:
    """Collect each encoder layer's output (the residual stream) while armed."""

    def __init__(self, model):
        self.outputs = None
        for layer in model.encoder.layers:
            layer.register_forward_hook(self._record)

    def _record(self, _module, _inputs, out):
        if self.outputs is not None:
            self.outputs.append(out[0] if isinstance(out, tuple) else out)

    def __enter__(self):
        self.outputs = []
        return self.outputs

    def __exit__(self, *_exc):
        self.outputs = None


@torch.no_grad()
def predict(model, x, device, batch_size=64):
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(np.asarray(x[i:i + batch_size], dtype=np.float32)).to(device)
        out.append(model(xb)["logits"].float().reshape(-1).cpu())
    return torch.cat(out).numpy()


def summarize(prob, labels, reference=None):
    s = {"accuracy": float(((prob > 0.5) == labels).mean())}
    if reference is not None:
        s["agreement_with_original"] = float(((prob > 0.5) == (reference > 0.5)).mean())
        s["prob_mean_abs_diff"] = float(np.abs(prob - reference).mean())
    return s


def to_np32(a):
    return np.asarray(a, dtype=np.float32)


# --------------------------------------------------------------------------
# Quantization-aware fine-tuning

class ActivationFakeQuant(nn.Module):
    """Per-tensor asymmetric int8 fake quantization, as rknn uses for activations.

    The range follows an exponential moving average of each training batch's
    min/max. Gradients pass straight through the rounding.
    """

    def __init__(self, momentum=0.01):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("lo", torch.tensor(0.0))
        self.register_buffer("hi", torch.tensor(0.0))
        self.register_buffer("seeded", torch.tensor(False))

    def forward(self, x):
        if self.training:
            lo, hi = x.detach().min(), x.detach().max()
            if self.seeded:
                self.lo.lerp_(lo, self.momentum)
                self.hi.lerp_(hi, self.momentum)
            else:
                self.lo.copy_(lo)
                self.hi.copy_(hi)
                self.seeded.fill_(True)
        lo, hi = self.lo.clamp(max=0), self.hi.clamp(min=0)  # keep 0 representable
        scale = (hi - lo).clamp(min=1e-8) / 255
        zero_point = torch.round(-lo / scale) - 128
        q = torch.clamp(torch.round(x / scale) + zero_point, -128, 127)
        return x + ((q - zero_point) * scale - x).detach()


class WeightFakeQuant(nn.Module):
    """Per-output-channel symmetric int8 rounding (rknn's 'channel' method)."""

    def forward(self, w):
        scale = w.detach().abs().amax(dim=tuple(range(1, w.dim())), keepdim=True).clamp(min=1e-8) / 127
        return w + (torch.clamp(torch.round(w / scale), -127, 127) * scale - w).detach()


class QuantizationSimulator:
    """Simulate int8 inference during fine-tuning.

    Weights of every Linear/Conv1d are rounded per channel. Activations are
    fake-quantized at the inputs and outputs of those layers and at the inputs
    of every DyT, which covers the tensors rknn stores as int8 between NPU ops.
    """

    def __init__(self, model):
        from torch.nn.utils import parametrize

        self._parametrize = parametrize
        self.weighted, self.quantized, self.handles = [], [], []
        device = next(model.parameters()).device
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                parametrize.register_parametrization(module, "weight", WeightFakeQuant())
                self.weighted.append(module)
                self._quantize_input(module, device)
                out_q = ActivationFakeQuant().to(device)
                module.qat_output = out_q
                self.handles.append(module.register_forward_hook(lambda _m, _a, out, q=out_q: q(out)))
            elif isinstance(module, DynamicTanh):
                self._quantize_input(module, device)

    def _quantize_input(self, module, device):
        in_q = ActivationFakeQuant().to(device)
        module.qat_input = in_q
        self.quantized.append(module)
        self.handles.append(module.register_forward_pre_hook(lambda _m, args, q=in_q: (q(args[0]),) + args[1:]))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        for module in self.weighted:
            self._parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
        for module in self.quantized:
            for attr in ("qat_input", "qat_output"):
                if hasattr(module, attr):
                    delattr(module, attr)


def cmd_train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    name = model_name(args.seconds)

    teacher = load_reference_model(args.onnx).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)

    # The teacher always sees the full 8-second window; the student sees the
    # last `seconds` of it (the same arrays when seconds == 8).
    x_eval, y_eval = load_eval(args.eval)
    xt_eval, yt_eval = load_eval(args.teacher_eval or args.eval)
    feats, labels = load_features(args.features)
    t_feats, t_labels = load_features(args.teacher_features or args.features)
    if not (np.array_equal(y_eval, yt_eval) and np.array_equal(labels, t_labels)):
        raise ValueError("student and teacher inputs are not aligned sample by sample")
    n = len(labels)
    print(f"training samples: {n} (positives {int(labels.sum())}), student window {args.seconds}s")

    # Check the recovered weights before building on them.
    session = ort.InferenceSession(args.onnx)
    p_onnx = np.concatenate([session.run(None, {"input_features": xt_eval[i:i + 1]})[0].reshape(-1)
                             for i in range(len(xt_eval))])
    p_teacher = predict(teacher, xt_eval, device)
    print(f"recovered weights: max |torch - onnx| = {np.abs(p_teacher - p_onnx).max():.2e}")
    print("original (fp32) " + json.dumps(summarize(p_teacher, y_eval)))

    student = shorten_window(load_reference_model(args.onnx).to(device), args.seconds)
    calib = torch.from_numpy(to_np32(feats[rng.choice(n, 64, replace=False)])).to(device)
    replace_with_dyt(student, calib, args.gain)
    if args.resume:
        student.load_state_dict(torch.load(args.resume, map_location=device))
    p_init = predict(student, x_eval, device)
    print(("resumed " if args.resume else "DyT before fine-tuning ") + json.dumps(summarize(p_init, y_eval, p_teacher)))

    bce = nn.BCEWithLogitsLoss()
    teacher_rec, student_rec = LayerOutputRecorder(teacher), LayerOutputRecorder(student)

    def fine_tune(epochs, lr, tag):
        optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
        steps = epochs * (n // args.batch_size)
        warmup = max(1, int(0.05 * steps))
        schedule = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
        step, t0 = 0, time.time()
        for epoch in range(epochs):
            order = rng.permutation(n)
            for i in range(0, n - args.batch_size + 1, args.batch_size):
                idx = np.sort(order[i:i + args.batch_size])
                xb = torch.from_numpy(to_np32(feats[idx])).to(device)
                xt = torch.from_numpy(to_np32(t_feats[idx])).to(device)
                yb = torch.from_numpy(labels[idx].astype(np.float32)).to(device)
                with torch.no_grad(), teacher_rec as teacher_hidden:
                    t_prob = teacher(xt)["logits"].float().reshape(-1)
                student.train()
                with student_rec as student_hidden:
                    s_logit = to_logit(student(xb)["logits"]).reshape(-1)
                # Output distillation against the original model's probability,
                # plus each layer's residual stream (over the positions the
                # student sees), scaled by its variance.
                distill = nn.functional.binary_cross_entropy_with_logits(s_logit, t_prob)
                hidden = sum(nn.functional.mse_loss(s, t[:, -s.shape[1]:]) / t.var()
                             for s, t in zip(student_hidden, teacher_hidden)) / N_LAYERS
                loss = distill + args.hidden_weight * hidden + args.label_weight * bce(s_logit, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                schedule.step()
                step += 1
                if step % 50 == 0:
                    print(f"{tag} epoch {epoch} step {step}/{steps} loss {loss.item():.4f} "
                          f"distill {distill.item():.4f} hidden {hidden.item():.4f} ({time.time() - t0:.0f}s)",
                          flush=True)
            p_student = predict(student, x_eval, device)
            print(f"epoch {epoch} {tag} " + json.dumps(summarize(p_student, y_eval, p_teacher)), flush=True)

    if args.epochs:
        fine_tune(args.epochs, args.lr, "DyT")
    if args.qat_epochs:
        name += "-qat"
        qat = QuantizationSimulator(student)
        with torch.no_grad():  # seed the activation ranges
            student.train()
            student(calib)
        fine_tune(args.qat_epochs, args.qat_lr, "DyT+QAT")
        qat.remove()  # bakes the int8-rounded weights into the model

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(student.state_dict(), os.path.join(args.out_dir, f"{name}.pt"))
    onnx_path = export_to_onnx_fp32(student.cpu().eval(), os.path.join(args.out_dir, f"{name}.onnx"),
                                    CONFIG, n_frames=100 * args.seconds)
    if not onnx_path:
        raise RuntimeError("ONNX export failed")
    summary = {"original_fp32": summarize(p_onnx, y_eval),
               "dyt_fp32": evaluate_onnx(onnx_path, x_eval, y_eval, p_onnx)}
    print("FINAL " + json.dumps(summary))
    with open(os.path.join(args.out_dir, f"{name}.eval.json"), "w") as f:
        json.dump({**summary, "args": {k: v for k, v in vars(args).items() if k != "func"}}, f, indent=2)


def evaluate_onnx(onnx_path, x, y, reference=None):
    session = ort.InferenceSession(onnx_path)
    prob = np.concatenate([session.run(None, {"input_features": x[i:i + 1]})[0].reshape(-1)
                           for i in range(len(x))])
    return summarize(prob, y, reference)


def cmd_evaluate(args):
    x, y = load_eval(args.eval)
    reference = None
    if args.reference:
        session = ort.InferenceSession(args.reference)
        reference = np.concatenate([session.run(None, {"input_features": x[i:i + 1]})[0].reshape(-1)
                                    for i in range(len(x))])
    print(json.dumps(evaluate_onnx(args.onnx, x, y, reference)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("features", help="precompute training features")
    p.add_argument("--parquet", nargs="+", required=True, help="smart-turn-data-v3.2-train shards")
    p.add_argument("--out", required=True, help="output prefix")
    p.add_argument("--seconds", type=int, default=8, help="input window length")
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("train", help="replace LayerNorm with DyT and fine-tune")
    p.add_argument("--onnx", required=True, help="original fp32 model (smart-turn-v3.2-gpu.onnx)")
    p.add_argument("--features", required=True, help="student input features (prefix from the features command)")
    p.add_argument("--eval", required=True, help="student eval.jsonl from prepare_data.py")
    p.add_argument("--seconds", type=int, default=8, help="student input window; must match --features")
    p.add_argument("--teacher-features", help="8-second features for the teacher (default: --features)")
    p.add_argument("--teacher-eval", help="8-second eval.jsonl for the teacher (default: --eval)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gain", type=float, default=4.0,
                   help="initial tanh input scale, in units of the typical token std")
    p.add_argument("--label-weight", type=float, default=0.5, help="weight of the ground-truth BCE term")
    p.add_argument("--hidden-weight", type=float, default=1.0,
                   help="weight of the per-layer residual-stream distillation term")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", help="start from a student checkpoint (.pt) written by an earlier run")
    p.add_argument("--qat-epochs", type=int, default=0,
                   help="quantization-aware epochs after the float epochs (output gets a -qat suffix)")
    p.add_argument("--qat-lr", type=float, default=2e-5)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="evaluate an ONNX model on eval.jsonl")
    p.add_argument("--onnx", required=True)
    p.add_argument("--eval", required=True)
    p.add_argument("--reference", help="original fp32 ONNX, to report agreement")
    p.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
