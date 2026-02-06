#!/usr/bin/env bash
# Benchmark: 3 configurations × 600 steps at H384
# RTX 4070 Laptop GPU (8GB), bf16 autocast
set -uo pipefail  # no -e: pretrain-mpnet exits 134 from a harmless PyGIL cleanup race

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

TOTAL_UPDATES=600
LOGGING_STEPS=10
BATCH_SIZE=16
MAX_TOKENS=512
WARMUP=60  # warmup updates (for LR schedule, also covers torch.compile warmup)

# Shared model config (H384, 8 layers)
MODEL_ARGS=(
    --dataset-name "HuggingFaceFW/fineweb-edu"
    --tokenizer-name "microsoft/mpnet-base"
    --encoder-layers 8
    --encoder-embed-dim 384
    --encoder-attention-heads 6
    --encoder-ffn-dim 1536
    --batch-size "$BATCH_SIZE"
    --total-updates "$TOTAL_UPDATES"
    --max-tokens "$MAX_TOKENS"
    --warmup-updates "$WARMUP"
    --logging-steps "$LOGGING_STEPS"
    --attention-dropout 0.0
    --update-freq 1
    --num-workers 0
)

gpu_peak_mem() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "N/A"
}

run_config() {
    local label="$1"; shift
    local logfile="$1"; shift
    local memfile="${logfile%.txt}_mem.txt"
    echo ""
    echo "$label"
    python -c "import torch; torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 2

    local start_time=$SECONDS
    pretrain-mpnet "$@" 2>&1 | tee "$logfile"
    local rc=${PIPESTATUS[0]}
    local elapsed=$((SECONDS - start_time))

    local gpu_mem
    gpu_mem=$(gpu_peak_mem)
    {
        echo "nvidia_smi_mem_mib=$gpu_mem"
        echo "elapsed_sec=$elapsed"
    } > "$memfile"
    echo "  => Elapsed: ${elapsed}s, GPU mem (nvidia-smi): ${gpu_mem} MiB"

    if [ "$rc" -ne 0 ] && [ "$rc" -ne 134 ]; then
        echo "ERROR: pretrain-mpnet exited with code $rc"
        return "$rc"
    fi
    return 0
}

echo "============================================================"
echo "  annotated-mpnet Benchmark Suite (H384, ${TOTAL_UPDATES} steps)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  Batch: ${BATCH_SIZE} × seq_len=${MAX_TOKENS}, update_freq=1"
echo "============================================================"

BENCH_START=$SECONDS

# ---------------------------------------------------------------------------
# Config 1: Baseline (no compile)
# ---------------------------------------------------------------------------
run_config "[1/3] Baseline (no compile)..." \
    "$SCRIPT_DIR/log_baseline_nocompile.txt" \
    "${MODEL_ARGS[@]}" \
    --checkpoint-dir "$SCRIPT_DIR/ckpt_baseline_nocompile"

# ---------------------------------------------------------------------------
# Config 2: Baseline (compiled)
# ---------------------------------------------------------------------------
run_config "[2/3] Baseline (compiled)..." \
    "$SCRIPT_DIR/log_baseline_compile.txt" \
    "${MODEL_ARGS[@]}" \
    --compile \
    --checkpoint-dir "$SCRIPT_DIR/ckpt_baseline_compile"

# ---------------------------------------------------------------------------
# Config 3: RoPE + FlexAttention (compiled)
# ---------------------------------------------------------------------------
run_config "[3/3] RoPE + FlexAttention (compiled)..." \
    "$SCRIPT_DIR/log_rope_flex_compile.txt" \
    "${MODEL_ARGS[@]}" \
    --use-rope \
    --no-relative-attention-bias \
    --compile \
    --flex-compile-block-mask \
    --checkpoint-dir "$SCRIPT_DIR/ckpt_rope_flex_compile"

echo ""
echo "All 3 runs complete in $((SECONDS - BENCH_START))s total."
echo "Analyzing..."
python "$SCRIPT_DIR/analyze_benchmark.py"
