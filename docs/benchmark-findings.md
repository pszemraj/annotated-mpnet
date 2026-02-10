# Benchmark Findings (H384 + H768)

This document summarizes benchmark outcomes collected from local runs in `benchmarks/runs/`.

## Scope

All runs below use:

- Streaming dataset: `HuggingFaceFW/fineweb-edu`
- Tokenizer: `microsoft/mpnet-base`
- `--dropout 0.0 --attention-dropout 0.0 --activation-dropout 0.0`
- Compiled runs where indicated

Primary run groups:

- H384 core suite: `benchmarks/runs/20260209_matrix_core`
- H768 core suite: `benchmarks/runs/20260210_h768_core_nodropout`
- H768 block-256 add-on attempts: `benchmarks/runs/20260210_h768_block256_addon`

## Key Outcome

Across successful runs, `RoPE + SDPA (compiled)` is the fastest configuration at each tested sequence length.

## H768 Results (Primary)

From `benchmarks/runs/20260210_h768_core_nodropout/reports/metrics.csv`:

### `seq=512`

| Config | Steady tokens/s | Relative to SDPA |
| ------ | --------------- | ---------------- |
| RoPE + SDPA | 57,400 | 1.000x |
| RoPE + Flex (block=128) | 49,682 | 0.866x |
| RoPE + Flex (block=256) | 49,225 | 0.858x |
| Baseline compiled | 46,472 | 0.810x |
| Baseline no compile | 36,535 | 0.636x |

### `seq=1024`

| Config | Steady tokens/s | Relative to SDPA |
| ------ | --------------- | ---------------- |
| RoPE + SDPA | 46,975 | 1.000x |
| RoPE + Flex (block=128) | 39,743 | 0.846x |

### `seq=2048`

| Config | Steady tokens/s | Relative to SDPA |
| ------ | --------------- | ---------------- |
| RoPE + SDPA | 36,847 | 1.000x |
| RoPE + Flex (block=128) | 33,178 | 0.900x |

## Additional H768 Attempts

Requested add-on tests (`block=256` at `seq=1024` and `seq=2048`) were attempted but not completed.

Observed failure modes:

- `rope_flex_b256_compile_1024_retry`: failed (OOM during compile/runtime setup)
- `rope_flex_b256_compile_2048_retry`: failed (OOM during compile/runtime setup)

Important context from OOM diagnostics:

- Another process occupied ~25 GiB VRAM during these attempts, leaving ~80 MB free in failing windows.
- This means these two add-on results are **inconclusive** for performance ranking.

Run artifacts:

- `benchmarks/runs/20260210_h768_block256_addon/logs/`
- `benchmarks/runs/20260210_h768_block256_addon/reports/metrics.csv`

## Known Inconclusive/Failed Configs

- H768 `seq=512`, Flex `block=64` (`rope_flex_b64_compile_512`) failed with a `torch._inductor` lowering assertion in Flex decoding path.
- H768 `seq=1024/2048`, Flex `block=256` add-on attempts failed under external VRAM pressure.

## Cross-Run Consistency Check (H384)

The H384 core run (`benchmarks/runs/20260209_matrix_core`) shows the same winner:

- `seq=512`: `rope_sdpa_compile_512` best
- `seq=1024`: `rope_sdpa_compile_1024` best
- `seq=2048`: `rope_sdpa_compile_2048` best

This supports the same conclusion as H768 on the current software stack.

## Practical Recommendation

On the current environment (`torch 2.9.1+cu128`), use:

- `--use-rope --no-relative-attention-bias --no-flex-attention --compile`

for best throughput among tested successful configurations.
