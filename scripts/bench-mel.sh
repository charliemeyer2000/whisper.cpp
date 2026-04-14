#!/usr/bin/env bash
#
# Build and run whisper-bench-mel against a fixture audio file.
#
# Usage:
#     scripts/bench-mel.sh                                # defaults: tiny model, samples/jfk.wav, 200 iters
#     scripts/bench-mel.sh -m models/ggml-base.bin -n 500
#     scripts/bench-mel.sh -f path/to/audio.wav -l candidate
#
# Emits the tool's JSON summary to stdout (mergeable into CI artifacts)
# and the human-readable table to stderr. To A/B two refs, run this
# script on each ref and diff the stdout JSON.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL=""
AUDIO="${REPO_ROOT}/samples/jfk.wav"
ITERATIONS=200
WARMUP=10
THREADS=""
LABEL=""
BUILD_DIR="${REPO_ROOT}/build-bench"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model) MODEL="$2"; shift 2 ;;
        -f|--file) AUDIO="$2"; shift 2 ;;
        -n|--iterations) ITERATIONS="$2"; shift 2 ;;
        -w|--warmup) WARMUP="$2"; shift 2 ;;
        -t|--threads) THREADS="$2"; shift 2 ;;
        -l|--label) LABEL="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,12p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${MODEL}" ]]; then
    # tiny is plenty — mel filter weights are model-size-independent.
    MODEL="${REPO_ROOT}/models/ggml-tiny.bin"
    if [[ ! -f "${MODEL}" ]]; then
        echo "==> downloading ggml-tiny model (only the mel filter weights are used)" >&2
        bash "${REPO_ROOT}/models/download-ggml-model.sh" tiny >&2
    fi
fi

if [[ ! -f "${MODEL}" ]]; then
    echo "error: model not found: ${MODEL}" >&2; exit 1
fi
if [[ ! -f "${AUDIO}" ]]; then
    echo "error: audio not found: ${AUDIO}" >&2; exit 1
fi

BENCH_BIN="${BUILD_DIR}/bin/whisper-bench-mel"
if [[ ! -x "${BENCH_BIN}" ]]; then
    echo "==> building whisper-bench-mel (Release, no Metal — CPU-only mel path)" >&2
    cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF >&2
    cmake --build "${BUILD_DIR}" --target whisper-bench-mel -j >&2
fi

ARGS=(-m "${MODEL}" -f "${AUDIO}" -n "${ITERATIONS}" -w "${WARMUP}")
if [[ -n "${THREADS}" ]]; then ARGS+=(-t "${THREADS}"); fi
if [[ -n "${LABEL}" ]];   then ARGS+=(-l "${LABEL}"); fi

"${BENCH_BIN}" "${ARGS[@]}"
