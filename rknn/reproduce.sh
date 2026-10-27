#!/bin/sh
# Reproduce the RKNN models in rknn/models/ from public inputs.
#
#   rknn/reproduce.sh            # run every step (each one skips finished work)
#
# Environment:
#   CACHE        downloads and features   (default: ~/.cache/smart-turn-rknn, ~8 GB)
#   OUT          models and reports       (default: rknn/build/repro; must be inside the repo)
#   PYTHON       interpreter with rknn/requirements-train.txt installed (default: python3)
#   HF_ENDPOINT  Hugging Face endpoint    (default: https://huggingface.co; e.g. https://hf-mirror.com)
#
# Needs Docker (the conversion runs in a linux/amd64 container) and, for the
# fine-tuning step, a GPU (CUDA or Apple MPS) for reasonable speed. Measured on
# an Apple M4 Pro: training ~13 min, the mmse conversions ~1 h.
#
# Outputs in $OUT:
#   smart-turn-v3.2-gpu-rv1106-int8-mmse.rknn       original weights, 8 s window
#   dyt-4s/smart-turn-v3.2-dyt-4s.onnx              fine-tuned fp32 model, 4 s window
#   dyt-4s/smart-turn-v3.2-dyt-4s-rv1106-int8-mmse.rknn
# plus *.eval.json accuracy summaries. Training on a GPU is not bit-for-bit
# deterministic, so a retrained model differs slightly from the committed one.
set -eu

cd "$(dirname "$0")/.."
CACHE=${CACHE:-$HOME/.cache/smart-turn-rknn}
OUT=${OUT:-rknn/build/repro}
PYTHON=${PYTHON:-python3}
HF=${HF_ENDPOINT:-https://huggingface.co}
export HF_ENDPOINT=$HF # transformers downloads the whisper-tiny config

# Pinned inputs: Hugging Face revisions and SHA-256 of each file.
MODEL_REV=f766f81d3cfdf7737ac64aad813d91bbfd56bf93 # pipecat-ai/smart-turn-v3
TEST_REV=dfc179241fee65b0efe81ca37786f41ce26f38ec  # pipecat-ai/smart-turn-data-v3.2-test
TRAIN_REV=f8c95d284595defb6b88a715c356cbc5c31b7ad1 # pipecat-ai/smart-turn-data-v3.2-train
TRAIN_SHARDS="00000 00013 00027 00041 00055 00069"     # 6 of the 83 training shards

log() { printf '\n== %s\n' "$*"; }
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

fetch() { # url dest sha256
    if [ ! -f "$2" ] || [ "$(sha256 "$2")" != "$3" ]; then
        echo "downloading $(basename "$2")"
        curl -fSL --retry 10 -C - -o "$2" "$1" || curl -fSL --retry 10 -o "$2" "$1"
    fi
    [ "$(sha256 "$2")" = "$3" ] || { echo "checksum mismatch: $2" >&2; exit 1; }
}

log "1/6 Download inputs to $CACHE"
mkdir -p "$CACHE/train" "$OUT"
fetch "$HF/pipecat-ai/smart-turn-v3/resolve/$MODEL_REV/smart-turn-v3.2-gpu.onnx" \
    "$CACHE/smart-turn-v3.2-gpu.onnx" ab8dc64b88713f90b571c15b714bd1330e6c883cad8763dacf65c9376dc539be
fetch "$HF/datasets/pipecat-ai/smart-turn-data-v3.2-test/resolve/$TEST_REV/data/train-00000-of-00010.parquet" \
    "$CACHE/test-00000.parquet" a87c75806b814ee7379998b6f9dc65a6433c01bfec2875e62c5d1ccd2b37257a
for shard in $TRAIN_SHARDS; do
    case $shard in
        00000) sum=f0d0b1c816019c0e9694cd3b36f5d9c509a4ad0fb861e3510a42956c84eb5e58 ;;
        00013) sum=e896e8d796b59af21ad931232ed141f7b12e1b423dfb15e55b4578d8f74701ee ;;
        00027) sum=6677ec9c0d54b6a26adbfb27953143d8fc9989f17a91c448937512db7d45e7d7 ;;
        00041) sum=99816c87c9fd9972faa9957cb95739f97d1d61d2beb8827e4ddb7c7e82932322 ;;
        00055) sum=1dcdf16e5732fe73b4eab1e3943c94f5433a2605a0c3e65d75c408cbf7942aa3 ;;
        00069) sum=68b3e9f9902c2332bf2458b0f309c9c75c3b7afd2ec66400b212afd001759152 ;;
    esac
    fetch "$HF/datasets/pipecat-ai/smart-turn-data-v3.2-train/resolve/$TRAIN_REV/data/train-$shard-of-00083.parquet" \
        "$CACHE/train/train-$shard-of-00083.parquet" "$sum"
done

log "2/6 Build the conversion image"
docker build --platform linux/amd64 -q -t smart-turn-rknn rknn >/dev/null
RUN="docker run --rm --platform linux/amd64 -v $PWD:/work -v $CACHE:/data smart-turn-rknn"

log "3/6 Calibration and evaluation features (8 s and 4 s windows)"
# 300 class-balanced calibration samples and 1000 disjoint evaluation samples
# from test shard 0; the same recordings for both windows.
[ -f "$CACHE/features/eval.jsonl" ] ||
    $RUN python rknn/prepare_data.py --parquet /data/test-00000.parquet --out /data/features
[ -f "$CACHE/features-4s/eval.jsonl" ] ||
    $RUN python rknn/prepare_data.py --parquet /data/test-00000.parquet --out /data/features-4s --seconds 4

log "4/6 Training features (teacher: 8 s, student: 4 s)"
shards=""
for shard in $TRAIN_SHARDS; do shards="$shards $CACHE/train/train-$shard-of-00083.parquet"; done
# shellcheck disable=SC2086 # $shards is a list of paths
[ -f "$CACHE/train/feats.count" ] ||
    "$PYTHON" rknn/replace_layernorm.py features --parquet $shards --out "$CACHE/train/feats"
# shellcheck disable=SC2086
[ -f "$CACHE/train/feats-4s.count" ] ||
    "$PYTHON" rknn/replace_layernorm.py features --parquet $shards --out "$CACHE/train/feats-4s" --seconds 4

log "5/6 Fine-tune the LayerNorm-free (DyT) 4 s model"
[ -f "$OUT/dyt-4s/smart-turn-v3.2-dyt-4s.onnx" ] ||
    "$PYTHON" rknn/replace_layernorm.py train --onnx "$CACHE/smart-turn-v3.2-gpu.onnx" --seconds 4 \
        --features "$CACHE/train/feats-4s" --teacher-features "$CACHE/train/feats" \
        --eval "$CACHE/features-4s/eval.jsonl" --teacher-eval "$CACHE/features/eval.jsonl" \
        --out-dir "$OUT/dyt-4s" --epochs 6 --seed 0

log "6/6 Convert to RKNN (rv1106, int8, mmse) and evaluate in the simulator"
[ -f "$OUT/smart-turn-v3.2-gpu-rv1106-int8-mmse.eval.json" ] ||
    $RUN python rknn/convert.py --onnx /data/smart-turn-v3.2-gpu.onnx --calib /data/features/calib.txt \
        --out-dir "$OUT" --algorithm mmse --eval /data/features/eval.jsonl
[ -f "$OUT/dyt-4s/smart-turn-v3.2-dyt-4s-rv1106-int8-mmse.eval.json" ] ||
    $RUN python rknn/convert.py --onnx "$OUT/dyt-4s/smart-turn-v3.2-dyt-4s.onnx" \
        --calib /data/features-4s/calib.txt --out-dir "$OUT/dyt-4s" --algorithm mmse \
        --eval /data/features-4s/eval.jsonl

log "Done"
for f in "$OUT/smart-turn-v3.2-gpu-rv1106-int8-mmse" "$OUT/dyt-4s/smart-turn-v3.2-dyt-4s-rv1106-int8-mmse"; do
    echo "$f.rknn"
    cat "$f.eval.json"
    echo
done
