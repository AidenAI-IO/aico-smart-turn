# Smart Turn v3 on Rockchip NPU (RKNN)

Smart Turn v3.2 converted to int8 `.rknn` models for the RV1103/RV1106 NPU,
the scripts that rebuild them from public inputs, and C code that runs them on
the board.

- [Models](#models): what is committed and how to use it
- [Reproduce](#reproduce): rebuild every model from scratch
- [Running on the board](#running-on-the-board): C library, helper, benchmark
- [Results and design notes](#results-and-design-notes): measurements and the
  path that led to these models

## Models

`models/` holds the board-validated builds (checksums in `models/SHA256SUMS`):

| File | Window | What it is |
|---|---|---|
| `smart-turn-v3.2-dyt-4s-rv1106-int8-mmse.rknn` | 4 s | **Recommended.** LayerNorm-free (DyT) model fine-tuned from v3.2, int8, `mmse` calibration |
| `smart-turn-v3.2-dyt-4s.onnx` | 4 s | The fine-tuned fp32 model the `.rknn` above was converted from |
| `smart-turn-v3.2-gpu-rv1106-int8-mmse.rknn` | 8 s | The released v3.2 weights, unchanged, int8, `mmse` calibration (baseline) |
| `*.eval.json` | | Accuracy summaries written when each model was built |

All numbers below come from the same 1000 held-out samples (shard 0 of
`pipecat-ai/smart-turn-data-v3.2-test`), measured on an RV1106 board.

| Model | Accuracy (fp32 / simulator / board) | Board `rknn_run` | CPU time per inference | NPU memory |
|---|---|---|---|---|
| Original v3.2, fp32 ONNX | 93.2% / — / — | — | — | — |
| `smart-turn-v3.2-gpu-rv1106-int8-mmse.rknn` | 93.2% / 92.8% / 92.9% | 516 ms | 278 ms | 11.7 MB |
| **`smart-turn-v3.2-dyt-4s-rv1106-int8-mmse.rknn`** | 92.8% / 92.8% / 93.1% | **72.5 ms** | 6 ms | 9.8 MB |

Model interface:

- Input `input_features`: Whisper log-mel features, shape `(1, 80, frames)`.
  Use 400 frames for the 4 s model and 800 for the 8 s model. Compute them
  from the last 4 s or 8 s of 16 kHz audio, zero-padded at the front, with
  `WhisperFeatureExtractor(chunk_length=4 or 8)` and `do_normalize=True`, or
  with `board/whisper_mel.c`. The 4 s model therefore differs from
  `inference.py`, which uses 8 s.
- Output `logits`: shape `(1, 1)`, the probability that the turn is complete
  (sigmoid already applied). Treat `> 0.5` as complete.
- RV1106 models are "mini RKNN" models with **int8** input and output. Query
  `RKNN_QUERY_NATIVE_INPUT_ATTR` / `..._OUTPUT_ATTR` for `scale` and `zp`.
  Quantize as `round(x / scale) + zp`, and dequantize as `(q - zp) * scale`.
  `board/smart_turn.c` does all of this.
- Runtime: `librknnmrt.so` 2.5.0, matching rknn-toolkit2 2.5.0.

## Reproduce

`reproduce.sh` rebuilds all three models. It downloads the pinned inputs and
verifies their checksums, computes features, fine-tunes the 4 s DyT model,
converts both models, and evaluates them in the rknn-toolkit2 simulator.
Each step skips work that is already done, so an interrupted run can be
restarted.

```bash
# Training environment (Python 3.12); conversion runs in Docker
python3.12 -m venv ~/venvs/smart-turn-rknn
~/venvs/smart-turn-rknn/bin/pip install -r rknn/requirements-train.txt

PYTHON=~/venvs/smart-turn-rknn/bin/python rknn/reproduce.sh
# If huggingface.co is unreachable: HF_ENDPOINT=https://hf-mirror.com rknn/reproduce.sh
```

Requirements: Docker (the conversion image is `linux/amd64`; Docker Desktop on
Apple silicon emulates it), about 8 GB in `CACHE`, and a GPU (CUDA or Apple
MPS) for the fine-tuning step. On an Apple M4 Pro, fine-tuning takes about
13 minutes and the two `mmse` conversions about an hour. Outputs go to
`rknn/build/repro/`.

Pinned inputs (Hugging Face revisions; SHA-256 values are in `reproduce.sh`):

| Input | Source |
|---|---|
| `smart-turn-v3.2-gpu.onnx` (fp32 weights) | `pipecat-ai/smart-turn-v3` @ `f766f81d` |
| Calibration and evaluation audio: `data/train-00000-of-00010.parquet` | `pipecat-ai/smart-turn-data-v3.2-test` @ `dfc17924` |
| Fine-tuning audio: shards 0, 13, 27, 41, 55, 69 of 83 (19,587 samples) | `pipecat-ai/smart-turn-data-v3.2-train` @ `f8c95d28` |

Pinned environments:

- Conversion: `Dockerfile` (python:3.10-slim pinned by digest) and
  `requirements.txt` (rknn-toolkit2 2.5.0, onnx 1.18.0, onnxruntime 1.23.2,
  onnxsim 0.7.3, numpy 1.26.4, transformers 5.20.2).
- Fine-tuning: `requirements-train.txt` (torch 2.14.1, transformers 5.20.2,
  onnx 1.23.2, onnxscript 0.7.2).

Fine-tuning on a GPU is not bit-for-bit deterministic, so a retrained
`smart-turn-v3.2-dyt-4s.onnx` differs slightly from the committed one. To
rebuild the committed `.rknn` from the committed fp32 model, run step 6 below
with `--onnx rknn/models/smart-turn-v3.2-dyt-4s.onnx`. Whether rknn-toolkit2
produces byte-identical output across runs has not been checked; compare the
`.eval.json` summaries.

### What the script runs

`RUN` is the container command:
`docker run --rm --platform linux/amd64 -v $PWD:/work -v $CACHE:/data smart-turn-rknn`.

1. Download the inputs above into `$CACHE` and check their SHA-256.
2. `docker build --platform linux/amd64 -t smart-turn-rknn rknn`. Use the
   amd64 image even on Apple silicon: the aarch64 rknn-toolkit2 wheel tries to
   open a real NPU (`/dev/rknpu`) while compiling and crashes without one.
3. `prepare_data.py` (in the container) picks 300 class-balanced calibration
   samples and 1000 disjoint evaluation samples from the test shard, and
   writes their features for an 8 s window (`features/`) and a 4 s window
   (`features-4s/`). Both windows use the same recordings.
4. `replace_layernorm.py features` decodes the six training shards into 8 s
   features for the teacher and 4 s features for the student.
5. `replace_layernorm.py train --seconds 4` recovers the v3.2 weights from the
   fp32 ONNX file, replaces the encoder LayerNorms with DyT, shortens the
   window to 4 s, and fine-tunes by distillation (6 epochs, batch 32,
   lr 1e-4, seed 0). It then exports `smart-turn-v3.2-dyt-4s.onnx`.
6. `convert.py` (in the container) fixes the batch size to 1, simplifies the
   graph with onnxsim, quantizes to int8 (`w8a8`, per-channel, `mmse`, target
   `rv1106`), exports the `.rknn`, and compares it with the fp32 model in the
   simulator. It writes `<model>-rv1106-int8-mmse.rknn` and `.eval.json`.
   Options: `--algorithm normal|mmse|kl_divergence`, `--target`, and
   `--analyze <one calib .npy>` for rknn-toolkit2's per-layer error report.

Convert from fp32 models only. RKNN quantizes the model itself and cannot
reuse the QDQ graph of Pipecat's int8 `smart-turn-v3.2-cpu.onnx`.

Other scripts:

- `replace_layernorm.py train` without `--seconds/--teacher-*` trains the 8 s
  DyT model, and `--qat-epochs N` adds quantization-aware fine-tuning. Neither
  is used for the committed models; see the design notes.
- `replace_layernorm.py evaluate --onnx M --eval eval.jsonl --reference R`
  reports fp32 accuracy and agreement with a reference model.

## Running on the board

`board/` holds a small C library and two programs. `board/build.sh`
cross-compiles them for armhf glibc (e.g. Debian on RV1106) in a Debian
container, and downloads `librknnmrt.so` 2.5.0 next to the binaries, which
find it through an `$ORIGIN` rpath. For Buildroot/uclibc firmware, build
against `armhf-uclibc/librknnmrt.so` with the vendor's
`arm-rockchip830-linux-uclibcgnueabihf` toolchain instead.

| File | Purpose |
|---|---|
| `smart_turn.h/.c` | Open a model; audio (16 kHz int16) → log-mel → int8 → `rknn_run` → probability. The window (4 s or 8 s) is read from the model. |
| `whisper_mel.h/.c` | Whisper log-mel features in C (normalisation, reflect-padded STFT with a mixed-radix FFT, Slaney mel filters, log/clamp). Matches `WhisperFeatureExtractor` to within 5e-5. |
| `whisper_mel_filters.h` | Mel filter weights, generated from transformers by `gen_mel_filters.py`. |
| `smart_turn_helper.c` | Helper process for applications, in the style of the firmware's `rknn_vad`: stdin `T` + u32 sample count + int16 PCM, stdout `P <prob> <features_ms> <run_ms>`. |
| `smart_turn_bench.c` | Latency and memory, and accuracy with `--features` (precomputed float32 features) or `--audio` (raw PCM, full path). |
| `eval_audio.py` | Exports the 1000 evaluation recordings as PCM records plus labels, and scores `smart_turn_bench` output. |
| `simulate_turns.py` | Replays recordings through the firmware's VAD helper and Smart Turn (`record`, on the device, pure Python), then compares end-of-turn policies offline (`sweep`). |

```bash
rknn/board/build.sh
scp rknn/build/board/{smart_turn_helper,smart_turn_bench,librknnmrt.so} \
    rknn/models/smart-turn-v3.2-dyt-4s-rv1106-int8-mmse.rknn user@board:smart-turn/
ssh user@board 'cd smart-turn && ./smart_turn_bench smart-turn-v3.2-dyt-4s-rv1106-int8-mmse.rknn --iters 50'

# Accuracy of the full path (C features + NPU) on the 1000 evaluation recordings
python rknn/board/eval_audio.py export --parquet $CACHE/test-00000.parquet \
    --eval $CACHE/features/eval.jsonl --out eval1000
ssh user@board 'cd smart-turn && ./smart_turn_bench smart-turn-v3.2-dyt-4s-rv1106-int8-mmse.rknn \
    --audio - --count 1000 --iters 1' < eval1000.pcm > probs.txt
python rknn/board/eval_audio.py score --probs probs.txt --labels eval1000.labels
```

The user needs access to `/dev/rknpu`, which is usually granted through the
`video` group. `whisper_mel_filters.h` is regenerated with
`python rknn/board/gen_mel_filters.py > rknn/board/whisper_mel_filters.h`.

### Full path on the board (4-second DyT model)

The board: Luckfox-style RV1106 running Debian 13 armhf (glibc 2.41).
Cortex-A7 single core, 145 MB RAM for Linux plus a 100 MB CMA pool for the
NPU, rknpu driver 0.9.2, librknnmrt 2.5.0.

| | Result |
|---|---|
| Accuracy, raw audio → C features → NPU, 1000 samples | 93.0% (Python features on the board: 93.1%; same decision on 99.7%) |
| Features in C | 19.6 ms at full clock; 26–36 ms median with the `ondemand` governor, which lowers the clock while the CPU waits for the NPU |
| `rknn_run` | 72.5 ms |
| Helper call, real-time replay with the VAD running | median 102 ms, p90 111 ms |
| Process RSS | 1.8 MB (11.5 MB peak while loading the model) |
| `rknn_init` | 30–75 ms |

## Results and design notes

How the committed models came about, in order. Unless noted, accuracy uses
the 1000 evaluation samples, and RKNN numbers come from the simulator.

### 1. Converting the released weights (8 s window)

Evaluated on 1000 held-out samples from `pipecat-ai/smart-turn-data-v3.2-test`
(shard 0, disjoint from the 300 calibration samples). RKNN numbers come from
the rknn-toolkit2 simulator, not from a board.

| Model | Accuracy | Same decision as fp32 | Mean abs. prob. diff vs fp32 |
|---|---|---|---|
| fp32 ONNX (`smart-turn-v3.2-gpu.onnx`) | 93.2% | — | — |
| int8 ONNX shipped with Pipecat (`smart-turn-v3.2-cpu.onnx`) | 92.1% | 95.5% | 0.043 |
| RKNN int8, `normal` calibration | 90.7% | 93.7% | 0.072 |
| RKNN int8, `kl_divergence` calibration | 93.0% | 97.0% | 0.029 |
| **RKNN int8, `mmse` calibration (default)** | **92.8%** | **98.2%** | **0.025** |

Operator placement reported by the compiler:

- On the NPU: both Conv1d layers, every linear layer (QKV, output, FFN, pooling
  head, classifier), attention (fused into `exSDPAttention`), GELU, softmax, tanh, sigmoid.
- On the CPU: the 10 LayerNorms (`exNorm`), each over 400×384 values or fewer.

Memory reported by the compiler: 7.9 MB of weights and 1.8 MB of
intermediate tensors. The `.rknn` file is about 11 MB.

#### Measured on the board

Same board as above (Debian 13 armhf, glibc 2.41, Cortex-A7 single core,
145 MB RAM plus a 100 MB CMA pool for the NPU, rknpu driver 0.9.2,
librknnmrt 2.5.0). Measured with `board/smart_turn_bench`.

| | Result |
|---|---|
| `rknn_run` latency (50 runs) | median 516 ms (min 484, p90 531) |
| CPU time per inference | 278 ms (the runtime's CPU LayerNorm) |
| Input quantization (float → int8, in C) | 4 ms |
| NPU (DMA/CMA) memory | 11.7 MB |
| Process RSS | 1.8 MB steady, 12.7 MB peak while loading the model file |
| `rknn_init` | 35–75 ms |
| Accuracy vs fp32 (299 eval samples) | 97.7% same decision, mean abs. prob. diff 0.026 |

The on-board outputs match the simulator. Per-layer timings
(`RKNN_LOG_LEVEL=5`) show where the time goes:

| Part | Count | Each | Total |
|---|---|---|---|
| LayerNorm (`exNorm`, CPU) | 9 | ~26 ms | ~240 ms |
| Attention (`exSDPAttention`, NPU) | 4 | ~53 ms | ~213 ms |
| All Conv / linear layers (NPU) | | | ~20 ms |

Attempts to reduce this that did not help:

- `enable_flash_attention=True`: same `exSDPAttention` kernel and timing.
- Rewriting LayerNorm as ReduceMean/Sub/Mul/Sqrt/Div in the ONNX graph: the
  compiler fuses it back into a CPU `exNorm`.
- Forcing every `exNorm` onto the NPU with `op_target`. The keys must be the
  fused ops' output tensor names, e.g. `node_layer_norm#1` and `_v_393_tp_rs`.
  Latency dropped to 233 ms and CPU time to ~11 ms, and the simulator still
  agreed with fp32. On the board, however, the outputs were wrong: an almost
  constant ~0.49 probability, 52% accuracy, and a mean abs. prob. diff of 0.46.
  Upgrading the rknpu kernel driver from 0.9.2 to 0.9.8 (rockchip-linux/kernel
  `develop-5.10`, built as a module against the board's own kernel config) gave
  bit-identical outputs. The fault is in how the NPU executes the compiled
  LayerNorm, not in the kernel driver, so this build is not usable.
- librknnmrt has no custom-op support, so the CPU LayerNorm cannot be replaced.

### 2. Removing LayerNorm (DyT, 8 s window)

`replace_layernorm.py` replaces the encoder's nine LayerNorms with Dynamic
Tanh (`weight * tanh(alpha * (x - shift)) + bias`, Zhu et al. 2025). DyT is
element-wise, so it compiles to NPU Mul/Add/Tanh ops. The classifier head's
single LayerNorm (one 256-value vector, ~0.1 ms) stays on the CPU.

The student model starts from the released v3.2 weights. Only ONNX weights are
published, so the script recovers them from `smart-turn-v3.2-gpu.onnx`; they
reproduce its output to within 7e-6. The student is fine-tuned by distilling
the original model's probabilities and per-layer residual streams, plus the
ground-truth labels. Training used 19,587 samples (6 of the 83 train shards:
0, 13, 27, 41, 55, 69) for 6 epochs at batch size 32, about 21 minutes on an
Apple M4 Pro (MPS).

Results on the same 1000 evaluation samples. These samples were only
monitored during training; the reported model is simply the last epoch. RKNN
rows come from the simulator.

| Model | Accuracy | Same decision as its fp32 | Mean abs. prob. diff vs its fp32 |
|---|---|---|---|
| Original fp32 | 93.2% | — | — |
| DyT fp32 (before fine-tuning) | 70.6% | 72.2% vs original | 0.274 vs original |
| **DyT fp32** | **93.3%** | 96.1% vs original | 0.045 vs original |
| DyT RKNN int8, `normal` | 91.3% | 95.6% | 0.066 |
| DyT RKNN int8, `mmse` | 90.4% | 93.3% | 0.069 |
| Original RKNN int8, `mmse` (for comparison) | 92.8% | 98.2% | 0.025 |

In fp32, removing LayerNorm costs no accuracy. After int8 quantization, the
DyT model ends up about 1.5–2.5 points below the original model's RKNN build;
this loss comes from quantization, not from the architecture change.

On the board (same setup as above):

| | Original (`mmse`) | DyT (`normal`) | DyT (`mmse`) |
|---|---|---|---|
| `rknn_run` median | 516 ms | 241 ms | 236 ms |
| CPU time per inference | 278 ms | 10 ms | 11 ms |
| Accuracy, 300 eval samples | 92.3% | 90.3% | 91.3% |
| NPU memory | 11.7 MB | 11.8 MB | 11.8 MB |

The remaining time is almost all attention: 4 × ~54 ms in `exSDPAttention`.

### 3. Shortening the window to 4 s (the recommended model)

Attention cost grows with the square of the sequence length, so the next step
shortens the input window from 8 s to 4 s (200 encoder positions instead of
400). `replace_layernorm.py train --seconds 4` keeps the original model's
*last* 200 positional embeddings: inputs are aligned to the end of the audio,
so each position keeps its distance from the end of the speech. The teacher
still sees the full 8 seconds. Features for the student are computed from the
last 4 s of audio (`WhisperFeatureExtractor(chunk_length=4)`), as they would
be at deployment. Slicing 8-second features is not equivalent, because
Whisper's log-mel normalization uses the maximum over the whole window.

All rows use the same 1000 evaluation samples. Board rows ran all 1000
samples on the RV1106.

| Model | fp32 | RKNN simulator | Board | vs its own fp32 (board) | Board latency |
|---|---|---|---|---|---|
| Original, 8 s, `mmse` | 93.2% | 92.8% | 92.9% | 97.3% same decision, diff 0.028 | 516 ms |
| DyT, 4 s, `normal` | 92.8% | 92.4% | 91.7% | 96.7%, 0.042 | 72.5 ms |
| DyT, 4 s, + QAT, `normal` | 93.3% | 91.4% | 91.4% | 96.1%, 0.046 | 72.5 ms |
| **DyT, 4 s, `mmse`** | **92.8%** | **92.8%** | **93.1%** | **98.7%, 0.020** | **72.5 ms** |

The 4-second `mmse` model matches the original on the board. Of the 1000
samples, it gets 23 right that the original gets wrong, and the reverse
happens 21 times. Per-language accuracy moves by a few points in both
directions (30–50 samples per language, so within noise); Chinese is 83.9% for
both models. On the board it runs in 72.5 ms (`rknn_run`), 78.7 ms including
input quantization, with 6 ms of CPU time, 9.8 MB of NPU memory, and 1.7 MB of
process RSS. The remaining cost is still mostly attention (4 × 15 ms).

`--qat-epochs` adds quantization-aware fine-tuning: per-channel int8 weights,
plus per-tensor int8 activations around every Linear/Conv and DyT. It did not
help after conversion, most likely because rknn quantizes different tensors
(after op fusion) and picks its own ranges. `mmse` calibration alone closed
the gap.

### 4. End-of-turn policies with the firmware's VAD

`simulate_turns.py` replayed the 1000 evaluation recordings, each followed by
3.5 s of silence, through the firmware's Silero VAD helper (`rknn_vad`,
threshold 0.5, minimum speech 300 ms). Three policies ran on the same VAD
output:

- the firmware's current rule: end the turn after 550 ms of silence;
- Smart Turn: after 200 ms of silence, end the turn if Smart Turn says
  complete, otherwise wait (re-asking at each new pause) up to 3 s of
  silence. This is the pipecat default policy; it ran once with each model.

| Policy | Ended before the recording's speech was over | Latency after speech end, complete turns | Incomplete turns kept open for 3 s |
|---|---|---|---|
| 550 ms silence | 83 | 576 ms | 0 / 464 |
| Smart Turn, original 8 s model | 318 | 798 ms (median) | 289 / 346 (84%) |
| Smart Turn, 4 s DyT model | 341 | 319 ms (median), 320 ms (p90) | 276 / 329 (84%) |

Latency is measured from the end of the last speech frame. In a real-time
replay of 100 recordings, it was 325 ms median and 332 ms p90. During that
replay the whole system was 26% busy: the VAD helper used 20% of the core,
and the Smart Turn helper 0.5%.

Smart Turn holds 84% of the incomplete turns open, where the silence rule
ends all of them after 550 ms. It also answers complete turns 250 ms sooner.
However, with a 200 ms trigger it also ends many turns at pauses *inside* a
recording, typically 2–6 s before the recording's last speech. These look like
sentence or clause boundaries in multi-sentence recordings. The original 8 s
model does the same: 288 of these cuts are shared, 53 happen only with the
4 s model, and 30 only with the original. So this behaviour comes from Smart
Turn with an aggressive trigger, not from the RV1106 port. The test recordings
are TTS and human clips from the Smart Turn test set, not conversations
recorded on this device.

#### Parameter sweep (4-second DyT model)

`simulate_turns.py record` runs on the board. It runs the VAD once per
recording and calls Smart Turn at every pause that reaches 200, 300, 400, or
500 ms. `simulate_turns.py sweep` then replays those decisions offline for
every policy. `smart:200/3000/0.5` reproduces the online run above (342 vs.
341 cuts, same latency), and `silence:550` matches it exactly.

Policy `smart:<trigger>/<fallback>/<threshold>`: ask Smart Turn after
`<trigger>` ms of silence, end the turn if its probability exceeds
`<threshold>`, and otherwise end it at `<fallback>` ms of silence.
"Incomplete kept open" counts recordings that stop mid-thought where the turn
stayed open for at least 1.5 s, giving the speaker that long to continue.
Latency is for complete turns; the mean includes turns that waited for the
fallback.

| Policy | Cut before speech end | Latency median / p90 / mean | Complete turns that waited for the fallback | Incomplete kept open ≥ 1.5 s |
|---|---|---|---|---|
| silence:550 (current firmware) | 83 | 576 / 576 / 576 ms | 0 / 452 | 0 / 464 |
| silence:800 | 23 | 800 / 800 / 800 ms | 0 / 479 | 0 / 497 |
| silence:1000 | 11 | 1024 / 1024 / 1024 ms | 0 / 484 | 0 / 504 |
| smart:200/1500/0.5 | 342 | 319 / 320 / 338 ms | 5 / 328 | 276 / 329 |
| smart:300/1500/0.5 | 215 | 416 / 419 / 445 ms | 10 / 394 | 329 / 390 |
| smart:300/1500/0.7 | 194 | 417 / 423 / 485 ms | 25 / 400 | 357 / 405 |
| smart:400/1500/0.5 | 123 | 512 / 515 / 539 ms | 12 / 436 | 369 / 440 |
| smart:400/1500/0.7 | 106 | 512 / 516 / 560 ms | 21 / 439 | 396 / 454 |
| **smart:500/1500/0.5** | **75** | **609 / 612 / 641 ms** | 17 / 455 | **380 / 469 (81%)** |
| smart:500/1500/0.7 | 68 | 609 / 612 / 653 ms | 23 / 456 | 406 / 475 (85%) |

Findings:

- The trigger delay controls the trade-off. Each extra 100 ms of trigger
  adds about 100 ms of latency and removes 40–130 cuts. Pauses inside these
  recordings are mostly 200–500 ms long.
- The fallback (1.5 / 2 / 3 s) changes neither cuts nor median latency: few
  pauses inside a recording last over 1.5 s. A longer fallback only makes
  missed complete turns slower (mean latency +20–150 ms). Use 1.5 s.
- A higher threshold (0.7, 0.8) removes 10–30% of the cuts and keeps more
  incomplete turns open, at the cost of more complete turns waiting for the
  fallback.
- `smart:500/1500/0.5` beats the current 550 ms rule on cuts (75 vs. 83) at
  +33 ms median latency, and keeps 81% of incomplete turns open for 1.5 s
  (the silence rule keeps none open). `smart:400/1500/0.7` answers about
  100 ms sooner, with 106 cuts. A silence rule alone needs about 800 ms to
  cut fewer turns, and still keeps no incomplete turn open.

Commands. `eval1000.pcm` and `eval1000.labels` come from
`board/eval_audio.py export` (see [Running on the board](#running-on-the-board)).
One board pass took about 40 minutes for 1000 recordings.

```bash
# on the board
python3 simulate_turns.py record --audio eval1000.pcm --labels eval1000.labels \
  --smart-turn-helper ./smart_turn_helper \
  --smart-turn-model smart-turn-v3.2-dyt-4s-rv1106-int8-mmse.rknn --out rec.jsonl
# anywhere
python3 rknn/board/simulate_turns.py sweep --records rec.jsonl --keep-open-ms 1500
```

`--resume` continues an interrupted `record` run. The recording behind the
table above is committed as `results/simulate_turns-dyt-4s.jsonl` (Silero VAD
RKNN helper from the board firmware, threshold 0.5, minimum speech 300 ms), so
`sweep --records rknn/results/simulate_turns-dyt-4s.jsonl` reproduces the table
without a board.
