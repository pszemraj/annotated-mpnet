#!/usr/bin/env python3
"""Parse benchmark logs and produce comparison report."""

from __future__ import annotations

import ast
import csv
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

CONFIGS = {
    "baseline_nocompile": "Baseline (no compile)",
    "baseline_compile": "Baseline (compiled)",
    "rope_flex_compile": "RoPE+Flex (compiled)",
}

LOGGING_STEPS = 10  # must match run_benchmark.sh

# Steps to skip for throughput measurement (compile warmup)
WARMUP_STEPS = 100


def parse_log(path: Path, logging_steps: int = LOGGING_STEPS) -> list[dict]:
    """Extract per-logging-interval metrics from a training log.

    The training script logs dicts via ``LOGGER.info(dict)`` and ``rich`` wraps
    them across multiple lines, inserting ``pretrain_mpnet.py:NNNN`` markers.
    We join the full text, find ``{...}`` spans, strip the markers, collapse
    whitespace, and parse.  Step numbers are inferred from record order.
    """
    records = []
    text = path.read_text()
    # Find all {...} blocks (may span lines due to rich wrapping)
    for m in re.finditer(r"\{[^}]+\}", text, re.DOTALL):
        raw = m.group(0)
        if "'tts'" not in raw:
            continue
        # Strip rich source markers like "pretrain_mpnet.py:1894"
        cleaned = re.sub(r"\s*pretrain_mpnet\.py:\d+\s*", " ", raw)
        # Collapse all whitespace (newlines, multi-space)
        cleaned = re.sub(r"\s+", " ", cleaned)
        try:
            metrics = ast.literal_eval(cleaned)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(metrics, dict) or "tts" not in metrics:
            continue
        step = (len(records) + 1) * logging_steps
        metrics["step"] = step
        records.append(metrics)

    return records


def parse_mem_file(path: Path) -> dict:
    """Parse the _mem.txt sidecar file for peak memory and elapsed time."""
    result = {}
    if not path.exists():
        return result
    for line in path.read_text().splitlines():
        if "=" in line:
            key, val = line.split("=", 1)
            try:
                result[key.strip()] = float(val.strip())
            except ValueError:
                pass
    return result


def parse_peak_memory_from_log(path: Path) -> float | None:
    """Extract 'GPU peak memory: X.X MiB' from training log.

    Rich wraps the line across multiple lines, so we join all text and search.
    """
    text = re.sub(r"\s+", " ", path.read_text())
    m = re.search(r"GPU peak memory:\s*([\d.]+)\s*MiB", text)
    if m:
        return float(m.group(1))
    return None


def parse_gpu_util_file(path: Path) -> dict[str, float]:
    """Parse per-sample GPU utilization log written by run_benchmark.sh."""
    if not path.exists():
        return {}

    gpu_utils: list[float] = []
    mem_utils: list[float] = []
    mem_used: list[float] = []

    with open(path, newline="") as f:
        reader = csv.reader(f)
        # Skip header
        next(reader, None)
        for row in reader:
            if len(row) < 6:
                continue
            try:
                gpu_utils.append(float(row[3].strip()))
                mem_utils.append(float(row[4].strip()))
                mem_used.append(float(row[5].strip()))
            except ValueError:
                continue

    if not gpu_utils:
        return {}

    return {
        "gpu_samples": float(len(gpu_utils)),
        "avg_gpu_util_pct": sum(gpu_utils) / len(gpu_utils),
        "max_gpu_util_pct": max(gpu_utils),
        "avg_mem_util_pct": sum(mem_utils) / len(mem_utils) if mem_utils else 0.0,
        "avg_mem_used_mib": sum(mem_used) / len(mem_used) if mem_used else 0.0,
        "max_mem_used_mib": max(mem_used) if mem_used else 0.0,
    }


def main():
    results = {}
    for key, label in CONFIGS.items():
        log_path = SCRIPT_DIR / f"log_{key}.txt"
        if not log_path.exists():
            print(f"WARNING: {log_path} not found, skipping {label}")
            continue
        records = parse_log(log_path)
        if not records:
            print(f"WARNING: No metrics parsed from {log_path}")
            continue
        mem_path = SCRIPT_DIR / f"log_{key}_mem.txt"
        mem_info = parse_mem_file(mem_path)
        gpu_path = SCRIPT_DIR / f"log_{key}_gpu.csv"
        gpu_info = parse_gpu_util_file(gpu_path)
        peak_mem = (
            parse_peak_memory_from_log(log_path)
            or gpu_info.get("max_mem_used_mib")
            or mem_info.get("nvidia_smi_mem_mib")
        )
        results[key] = {
            "label": label,
            "records": records,
            "peak_mem_mib": peak_mem,
            "elapsed_sec": mem_info.get("elapsed_sec"),
            "gpu_info": gpu_info,
        }

    if not results:
        print("No results to analyze.")
        sys.exit(1)

    # --- Summary table ---
    print()
    print("=" * 80)
    print("  BENCHMARK RESULTS — H384, 8L, batch=16, max_tokens=512")
    print("=" * 80)
    print()

    header = (
        f"{'Configuration':<28} {'Avg TTS':>10} {'Steady TTS':>12} "
        f"{'Final Loss':>11} {'Peak Mem':>10} {'Avg GPU%':>9} {'Peak GPU%':>10} {'Elapsed':>9}"
    )
    print(header)
    print("-" * len(header))

    baseline_steady_tts = None

    for key in CONFIGS:
        if key not in results:
            continue
        r = results[key]
        records = r["records"]

        # All tts values
        all_tts = [rec["tts"] for rec in records if "tts" in rec]
        # Steady-state tts (after warmup steps)
        steady_records = [rec for rec in records if rec.get("step", 0) > WARMUP_STEPS]
        steady_tts = [rec["tts"] for rec in steady_records if "tts" in rec]

        avg_tts = sum(all_tts) / len(all_tts) if all_tts else 0
        avg_steady_tts = sum(steady_tts) / len(steady_tts) if steady_tts else 0
        final_loss = records[-1].get("loss", records[-1].get("sbal", "N/A"))
        peak_mem = f"{r['peak_mem_mib']:.0f} MiB" if r["peak_mem_mib"] else "N/A"
        gpu_info = r.get("gpu_info", {})
        avg_gpu = f"{gpu_info['avg_gpu_util_pct']:.1f}" if "avg_gpu_util_pct" in gpu_info else "N/A"
        peak_gpu = (
            f"{gpu_info['max_gpu_util_pct']:.1f}" if "max_gpu_util_pct" in gpu_info else "N/A"
        )
        elapsed = f"{r['elapsed_sec']:.0f}s" if r["elapsed_sec"] else "N/A"

        if key == "baseline_nocompile":
            baseline_steady_tts = avg_steady_tts

        print(
            f"{r['label']:<28} {avg_tts:>10.0f} {avg_steady_tts:>12.0f} "
            f"{final_loss:>11.4f} {peak_mem:>10} {avg_gpu:>9} {peak_gpu:>10} {elapsed:>9}"
        )

    # --- Speedup ratios ---
    print()
    print("Speedup vs Baseline (no compile) — steady-state tokens/sec:")
    print("-" * 50)
    for key in CONFIGS:
        if key not in results or baseline_steady_tts is None or baseline_steady_tts == 0:
            continue
        records = results[key]["records"]
        steady_records = [rec for rec in records if rec.get("step", 0) > WARMUP_STEPS]
        steady_tts = [rec["tts"] for rec in steady_records if "tts" in rec]
        avg = sum(steady_tts) / len(steady_tts) if steady_tts else 0
        speedup = avg / baseline_steady_tts
        print(f"  {results[key]['label']:<28} {speedup:.2f}x  ({avg:.0f} tok/s)")

    # --- Per-step CSV ---
    csv_path = SCRIPT_DIR / "benchmark_metrics.csv"
    with open(csv_path, "w") as f:
        f.write("config,step,loss,acc,tts,ttp,tpb\n")
        for key in CONFIGS:
            if key not in results:
                continue
            for rec in results[key]["records"]:
                step = rec.get("step", "")
                loss = rec.get("loss", rec.get("sbal", ""))
                acc = rec.get("acc", "")
                tts = rec.get("tts", "")
                ttp = rec.get("ttp", "")
                tpb = rec.get("tpb", "")
                f.write(f"{key},{step},{loss},{acc},{tts},{ttp},{tpb}\n")
    print()
    print(f"Per-step metrics saved to: {csv_path}")

    # --- Key ---
    print()
    print("Key: TTS=tokens/sec, Steady=after step 100 (compile warmup excluded)")
    print("GPU% columns are from nvidia-smi time-series logs captured during each run.")
    print()


if __name__ == "__main__":
    main()
