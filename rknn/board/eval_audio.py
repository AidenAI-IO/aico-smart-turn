"""Evaluation audio for the board programs, and scoring of their output.

  export  Write the evaluation recordings chosen by prepare_data.py as raw
          16 kHz int16 PCM records ([u32 n][int16 n] each), plus one 0/1
          label per line (1 = the recording ends at a complete turn). These
          are the inputs of `smart_turn_bench --audio` and
          `simulate_turns.py record`.
  score   Read `PROB <i> <p> ...` lines from smart_turn_bench and report
          accuracy against the labels.

    python rknn/board/eval_audio.py export --parquet $CACHE/test-00000.parquet \
        --eval $CACHE/features/eval.jsonl --out eval1000
    ssh board './smart_turn_bench MODEL.rknn --audio - --count 1000' < eval1000.pcm > probs.txt
    python rknn/board/eval_audio.py score --probs probs.txt --labels eval1000.labels

Pure Python apart from pyarrow/soundfile for `export`.
"""

import argparse
import io
import json
import struct


def cmd_export(args):
    import pyarrow.parquet as pq
    import soundfile as sf

    rows = [json.loads(line) for line in open(args.eval)]
    table = pq.read_table(args.parquet, columns=["id", "audio"])
    index = {k: i for i, k in enumerate(table.column("id").to_pylist())}
    audio = table.column("audio")
    with open(args.out + ".pcm", "wb") as pcm, open(args.out + ".labels", "w") as labels:
        for row in rows:
            data, rate = sf.read(io.BytesIO(audio[index[row["id"]]]["bytes"].as_py()), dtype="int16")
            if rate != 16000 or data.ndim != 1:
                raise ValueError(f"{row['id']}: expected 16 kHz mono, got {rate} Hz, shape {data.shape}")
            pcm.write(struct.pack("<I", len(data)))
            pcm.write(data.astype("<i2").tobytes())
            labels.write(f"{int(row['endpoint_bool'])}\n")
    print(f"wrote {len(rows)} recordings to {args.out}.pcm and {args.out}.labels")


def cmd_score(args):
    labels = [int(line) for line in open(args.labels) if line.strip()]
    probs = {}
    for line in open(args.probs):
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "PROB":
            probs[int(parts[1])] = float(parts[2])
    n = len(probs)
    correct = sum((p > args.threshold) == bool(labels[i]) for i, p in probs.items())
    print(f"{n} samples, accuracy {correct / n:.3f}" if n else "no PROB lines found")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export", help="write evaluation PCM records and labels")
    p.add_argument("--parquet", required=True, help="smart-turn-data-v3.2-test shard used by prepare_data.py")
    p.add_argument("--eval", required=True, help="eval.jsonl written by prepare_data.py")
    p.add_argument("--out", required=True, help="output prefix (.pcm and .labels are appended)")
    p.set_defaults(func=cmd_export)
    p = sub.add_parser("score", help="accuracy of smart_turn_bench PROB lines")
    p.add_argument("--probs", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.set_defaults(func=cmd_score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
