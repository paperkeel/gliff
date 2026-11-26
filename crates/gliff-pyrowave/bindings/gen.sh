#!/bin/sh
# Regenerate src/bindings.rs from the PyroWave C API header. Needs
# bindgen-cli (`cargo install bindgen-cli`), clang, the Vulkan headers and a
# PyroWave build (scripts/build-pyrowave.sh). The output is committed, so
# the crate builds without any of them.
#
# The functions are loaded at run time (bindgen --dynamic-loading), so
# gliff runs without libpyrowave-shared.so and falls back to H.264.
set -e
cd "$(dirname "$0")"
ROOT=../../..
bindgen wrapper.h \
    --dynamic-loading PyroWave \
    --dynamic-link-require-all \
    --allowlist-function 'pyrowave_.*' \
    --allowlist-type 'pyrowave_.*' \
    --allowlist-var 'PYROWAVE_.*' \
    --no-layout-tests \
    --no-doc-comments \
    --no-prepend-enum-name \
    --raw-line '#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, unsafe_op_in_unsafe_fn, clippy::all, clippy::undocumented_unsafe_blocks)]' \
    --output ../src/bindings.rs \
    -- -x c -std=c11 -I"$ROOT/target/pyrowave/include"
