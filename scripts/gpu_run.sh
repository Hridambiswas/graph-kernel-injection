#!/usr/bin/env bash
# One command to run everything on tomorrow's NVIDIA laptop.
#
#   bash scripts/gpu_run.sh           # full-size benchmark shapes
#   bash scripts/gpu_run.sh --small   # shapes trimmed for 8 GiB laptop GPU
#
# Behaviour:
#   - captures nvidia-smi + python/torch/triton/GPU details to env.txt
#   - runs verify-grads, verify-fwd, benchmark in that order
#   - every stage tees its raw stdout to results/<timestamp>/
#   - stops at the first failure (set -e) and leaves the log in place
#   - never pushes to git

set -euo pipefail

# -------- parse flags --------
MODE_FLAG=""
if [[ "${1:-}" == "--small" ]]; then
    MODE_FLAG="small"
    shift
fi

# -------- output directory --------
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="results/${TS}"
mkdir -p "${OUT}"

echo "[gpu_run] writing to ${OUT}"

# -------- environment capture --------
{
    echo "# gpu_run.sh environment capture"
    echo "timestamp_utc: ${TS}"
    echo "hostname: $(hostname)"
    echo "uname: $(uname -a)"
    echo ""
    echo "## nvidia-smi"
    nvidia-smi || echo "nvidia-smi failed"
    echo ""
    echo "## python / torch / triton"
    python --version
    python - <<'PY'
import torch
print("torch         :", torch.__version__)
print("torch.cuda    :", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count  :", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"gpu[{i}]       :", p.name, f"(SM {p.major}.{p.minor}, {p.total_memory // (1024**2)} MiB)")
try:
    import triton
    print("triton        :", triton.__version__)
except Exception as e:
    print("triton        : not importable:", type(e).__name__, e)
try:
    from torch.nn.attention.flex_attention import flex_attention  # noqa: F401
    print("flex_attention: available")
except Exception as e:
    print("flex_attention: not available:", type(e).__name__, e)
PY
} | tee "${OUT}/env.txt"

# -------- shape presets --------
if [[ "${MODE_FLAG}" == "small" ]]; then
    # Fits on an 8 GiB laptop GPU with headroom for Triton compile.
    BATCH=1
    HEADS=4
    SEQ_Q=1
    SEQ_LEN=16384
    HEAD_DIM=64
    SINK=4
    WINDOW=1024
    DTYPE=fp16
    WARMUP=10
    ITERS=25
else
    # Full-size defaults match HarnessConfig.
    BATCH=1
    HEADS=8
    SEQ_Q=1
    SEQ_LEN=131072
    HEAD_DIM=64
    SINK=4
    WINDOW=2048
    DTYPE=fp16
    WARMUP=25
    ITERS=100
fi

echo "[gpu_run] shapes: B=${BATCH} H=${HEADS} N_Q=${SEQ_Q} N_KV=${SEQ_LEN} D=${HEAD_DIM} sink=${SINK} window=${WINDOW} dtype=${DTYPE} warmup=${WARMUP} iters=${ITERS}" | tee -a "${OUT}/env.txt"

# -------- Stage 1: strict gradient verification --------
echo "[gpu_run] stage 1: verify-grads"
python benchmark.py verify-grads \
    --batch "${BATCH}" --heads "${HEADS}" --seq-q "${SEQ_Q}" \
    --seq-len "${SEQ_LEN}" --head-dim "${HEAD_DIM}" \
    --sink "${SINK}" --window "${WINDOW}" --dtype "${DTYPE}" \
    --warmup "${WARMUP}" --iters "${ITERS}" \
    2>&1 | tee "${OUT}/01_verify_grads.log"

# -------- Stage 2: forward correctness --------
echo "[gpu_run] stage 2: verify-fwd"
python benchmark.py verify-fwd \
    --batch "${BATCH}" --heads "${HEADS}" --seq-q "${SEQ_Q}" \
    --seq-len "${SEQ_LEN}" --head-dim "${HEAD_DIM}" \
    --sink "${SINK}" --window "${WINDOW}" --dtype "${DTYPE}" \
    --warmup "${WARMUP}" --iters "${ITERS}" \
    2>&1 | tee "${OUT}/02_verify_fwd.log"

# -------- Stage 3: performance sweep (fair baselines) --------
echo "[gpu_run] stage 3: benchmark"
python benchmark.py benchmark \
    --batch "${BATCH}" --heads "${HEADS}" --seq-q "${SEQ_Q}" \
    --seq-len "${SEQ_LEN}" --head-dim "${HEAD_DIM}" \
    --sink "${SINK}" --window "${WINDOW}" --dtype "${DTYPE}" \
    --warmup "${WARMUP}" --iters "${ITERS}" \
    2>&1 | tee "${OUT}/03_benchmark.log"

# -------- Stage 4: archive the CSV + any chrome traces --------
if [[ -f bench_results.csv ]]; then
    cp bench_results.csv "${OUT}/bench_results.csv"
fi
if [[ -d traces ]]; then
    cp -r traces "${OUT}/traces"
fi

echo "[gpu_run] done. Results in ${OUT}"
