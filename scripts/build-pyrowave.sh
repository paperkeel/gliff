#!/usr/bin/env bash
# Build libpyrowave-shared.so, the PyroWave C API, at pinned commits.
#
# PyroWave (https://github.com/Themaister/pyrowave, MIT) is the wavelet codec
# that gliff uses on direct LAN sessions. gliff loads the library at run
# time, so gliff builds and runs without it and falls back to H.264.
#
# Output: target/pyrowave/lib/libpyrowave-shared.so* and
# target/pyrowave/include/pyrowave/pyrowave.h. Needs git, cmake and a C++
# compiler. The sources are cloned into target/pyrowave/src once.
set -euo pipefail

PYROWAVE_COMMIT=a32b05e03d70826b19d4d11ff74fe166ed74371b
# The Granite commit that PyroWave's checkout_granite.sh pins.
GRANITE_COMMIT=53090429539bf3404250d18fe58db971f0907c04

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
OUT="$ROOT/target/pyrowave"
SRC="$OUT/src/pyrowave"
mkdir -p "$OUT/src"

fetch() { # dir url commit
    if [ ! -d "$1/.git" ]; then
        git init -q "$1"
        git -C "$1" remote add origin "$2"
    fi
    if [ "$(git -C "$1" rev-parse -q --verify HEAD 2>/dev/null)" != "$3" ]; then
        git -C "$1" fetch -q --depth 1 origin "$3"
        git -C "$1" checkout -q --detach FETCH_HEAD
    fi
}

fetch "$SRC" https://github.com/Themaister/pyrowave.git "$PYROWAVE_COMMIT"
fetch "$SRC/Granite" https://github.com/Themaister/Granite.git "$GRANITE_COMMIT"
for sub in third_party/volk third_party/khronos/vulkan-headers; do
    git -C "$SRC/Granite" submodule update -q --init --depth 1 "$sub"
done

cmake -S "$SRC" -B "$OUT/build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$OUT" -DCMAKE_INSTALL_LIBDIR=lib >/dev/null
cmake --build "$OUT/build" --target pyrowave-shared -j "$(nproc)"
mkdir -p "$OUT/lib" "$OUT/include/pyrowave"
cp -P "$OUT"/build/libpyrowave-shared.so* "$OUT/lib/"
cp "$SRC/pyrowave.h" "$OUT/include/pyrowave/"
echo "built $OUT/lib/libpyrowave-shared.so"
