#!/bin/sh
# Cross-compile the Smart Turn board programs for 32-bit ARM Linux with glibc
# (e.g. Debian or Ubuntu on RV1103/RV1106) inside an armhf Debian container.
#
# Output in rknn/build/board/:
#   smart_turn_helper  stdin/stdout helper for applications (see smart_turn_helper.c)
#   smart_turn_bench   latency, memory, and accuracy checks
#   librknnmrt.so      RKNN runtime, found next to the binaries via rpath
# For Buildroot/uclibc firmware, build against armhf-uclibc/librknnmrt.so with
# the vendor's arm-rockchip830-linux-uclibcgnueabihf toolchain instead.
set -eu

# librknnmrt 2.5.0 on master, matching rknn-toolkit2==2.5.0 in ../requirements.txt.
RUNTIME_URL=https://raw.githubusercontent.com/airockchip/rknn-toolkit2/master/rknpu2/runtime/Linux/librknn_api

cd "$(dirname "$0")/../.."
out=rknn/build/board
mkdir -p "$out"

fetch() {
    [ -s "$out/$(basename "$1")" ] || curl -fsSL --retry 5 -o "$out/$(basename "$1")" "$RUNTIME_URL/$1"
}
fetch armhf/librknnmrt.so
fetch include/rknn_api.h

src=rknn/board
lib="$src/smart_turn.c $src/whisper_mel.c"
cflags="-O2 -mcpu=cortex-a7 -mfpu=neon-vfpv4 -Wall -Wextra -I$src -I$out"
ldflags="-L$out -lrknnmrt -lm -Wl,-rpath,\\\$ORIGIN"

docker run --rm --platform linux/arm/v7 -v "$PWD":/work -w /work debian:trixie sh -c "
    apt-get update -qq && apt-get install -y -qq gcc libc6-dev >/dev/null &&
    gcc $cflags -o $out/smart_turn_bench $src/smart_turn_bench.c $lib $ldflags &&
    gcc $cflags -o $out/smart_turn_helper $src/smart_turn_helper.c $lib $ldflags
"
echo "built $out/smart_turn_bench $out/smart_turn_helper"
