"""Build calibration and evaluation feature sets for the RKNN conversion.

Reads one parquet shard of the Smart Turn test dataset
(pipecat-ai/smart-turn-data-v3.2-test), computes the same Whisper log-mel
features that inference.py feeds the ONNX model, and writes:

    <out>/calib/*.npy, <out>/calib.txt   class-balanced set used for int8 quantization
    <out>/eval/*.npy,  <out>/eval.jsonl  disjoint set used by evaluate.py

Each .npy holds one float32 input of shape (1, 80, 100 * seconds), i.e.
(1, 80, 800) for the default 8-second window.
"""

import argparse
import io
import json
import os
import random
import sys

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from transformers import WhisperFeatureExtractor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from audio_utils import truncate_audio_to_last_n_seconds  # noqa: E402

SAMPLE_RATE = 16000


def compute_features(extractor, audio, seconds=8):
    """Whisper log-mel features of the last `seconds` of audio (front-padded).

    `extractor` must be WhisperFeatureExtractor(chunk_length=seconds).
    """
    audio = truncate_audio_to_last_n_seconds(audio, n_seconds=seconds)
    inputs = extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="np",
        padding="max_length",
        max_length=seconds * SAMPLE_RATE,
        truncation=True,
        do_normalize=True,
    )
    return inputs.input_features.astype(np.float32)  # (1, 80, 100 * seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", required=True, help="Parquet shard of smart-turn-data-v3.2-test")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--calib-size", type=int, default=300)
    parser.add_argument("--eval-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seconds", type=int, default=8, help="Input window length")
    args = parser.parse_args()

    meta_cols = ["id", "language", "endpoint_bool", "dataset"]
    table = pq.read_table(args.parquet, columns=meta_cols)
    rows = table.to_pylist()
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)

    # Calibration: alternate complete/incomplete samples so both classes shape
    # the activation ranges. Evaluation: the next samples in shuffled order.
    pos = [i for i in order if rows[i]["endpoint_bool"]]
    neg = [i for i in order if not rows[i]["endpoint_bool"]]
    half = args.calib_size // 2
    calib = pos[:half] + neg[: args.calib_size - half]
    calib_set = set(calib)
    evaluation = [i for i in order if i not in calib_set][: args.eval_size]

    audio = pq.read_table(args.parquet, columns=["audio"]).column("audio")
    extractor = WhisperFeatureExtractor(chunk_length=args.seconds)

    def dump(split, indices):
        split_dir = os.path.join(args.out, split)
        os.makedirs(split_dir, exist_ok=True)
        written = []
        for i in indices:
            data, sr = sf.read(io.BytesIO(audio[i]["bytes"].as_py()), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != SAMPLE_RATE:
                print(f"skip {rows[i]['id']}: sample rate {sr}")
                continue
            path = os.path.join(split_dir, f"{rows[i]['id']}.npy")
            np.save(path, compute_features(extractor, data, args.seconds))
            written.append({**rows[i], "path": os.path.abspath(path)})
        return written

    calib_rows = dump("calib", calib)
    with open(os.path.join(args.out, "calib.txt"), "w") as f:
        f.writelines(r["path"] + "\n" for r in calib_rows)

    eval_rows = dump("eval", evaluation)
    with open(os.path.join(args.out, "eval.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in eval_rows)

    print(f"calib: {len(calib_rows)} samples, eval: {len(eval_rows)} samples "
          f"(eval positives: {sum(r['endpoint_bool'] for r in eval_rows)})")


if __name__ == "__main__":
    main()
