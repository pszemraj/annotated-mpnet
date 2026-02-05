# Development Guide

This document covers testing, development setup, and contributing to `annotated-mpnet`.

## Running Tests

```bash
# Run all tests
python -m unittest discover tests

# Run a specific test file
python -m unittest tests.test_data
python -m unittest tests.test_pretrain
python -m unittest tests.test_pretrain_smoke

# Run a specific test case
python -m unittest tests.test_pretrain.TestPretrainHelpers.test_weight_decay_grouping
```

## Test Suite Overview

### `tests/test_data.py` - Data Pipeline Tests

Tests for the data collation and streaming functionality. These run on CPU without CUDA.

| Test                                                          | What It Verifies                                                       |
| ------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `test_permuted_batch`                                         | Collator produces valid permuted batches with deterministic seeding    |
| `test_collator_skips_special_and_padding_targets`             | Targets exclude special tokens (CLS, SEP, PAD, MASK)                   |
| `test_collator_has_padding_false_for_equal_lengths`           | `has_padding` flag correctly tracks batch padding state                |
| `test_training_seeded_sampling`                               | `RandomSamplerWithSeed` produces deterministic, epoch-varied orderings |
| `test_streaming_dataset_skip_and_max`                         | `HFStreamingDataset` respects `skip_samples` and `max_samples`         |
| `test_streaming_dataset_defers_padding_to_collator`           | Streaming dataset does not pre-pad sequences                           |
| `test_collator_rng_state_roundtrip`                           | Collator RNG state save/restore produces identical batches             |
| `test_file_mode_resume_sampler_offset_preserves_collator_rng` | Resume with sampler offset preserves masking determinism               |
| `test_collator_falls_back_without_fast_perm`                  | Graceful fallback when Cython extension unavailable                    |
| `test_streaming_dataset_rng_state_roundtrip`                  | Streaming dataset RNG state is a no-op (by design)                     |

### `tests/test_pretrain.py` - Core Training Logic Tests

Unit tests for pretraining helper functions, model architecture, and training utilities. These run on CPU.

| Test                                              | What It Verifies                                             |
| ------------------------------------------------- | ------------------------------------------------------------ |
| `test_get_initial_best_loss`                      | Best-loss initialization fallback logic                      |
| `test_validate_tokenizer_vocab_size_*`            | Tokenizer/checkpoint vocab size validation                   |
| `test_weight_decay_grouping`                      | Biases and norm weights excluded from weight decay           |
| `test_polynomial_scheduler_step_indexing`         | LR scheduler warmup and decay behavior                       |
| `test_init_final_params_zeroes_padding_idx`       | Embedding padding_idx=0 zeroed during init                   |
| `test_encode_emb_handles_sinusoidal_positions`    | Sinusoidal positional embeddings work correctly              |
| `test_hf_max_positions_to_internal`               | HF max_positions conversion (subtracts 2 for CLS/SEP)        |
| `test_lm_head_weight_registered_and_tied`         | LM head weight tied to embedding weight                      |
| `test_sentence_encoder_gradient_checkpointing`    | Gradient checkpointing path runs without error               |
| `test_sentence_encoder_inner_states_layout`       | Inner states layout consistent across modes                  |
| `test_sentence_encoder_positions_*`               | Explicit positions follow default semantics                  |
| `test_two_stream_attention_padding_mask`          | Padding masks stay 2D; SDPA matches non-SDPA                 |
| `test_two_stream_mask_layout_matches_reference`   | Boolean mask layout matches legacy float construction        |
| `test_resolve_best_loss_*`                        | Best-loss resolution from various checkpoint sources         |
| `test_get_resume_metadata_*`                      | Resume metadata extraction for legacy vs. modern checkpoints |
| `test_select_resume_checkpoint_path_*`            | Resume checkpoint selection logic                            |
| `test_model_summary_avoids_double_counting`       | Parameter counting doesn't double-count nested modules       |
| `test_pretraining_forward_return_mlm`             | `return_mlm=True` yields logits with expected shapes         |
| `test_pretraining_gradient_checkpointing_forward` | Pretraining forward runs with gradient checkpointing         |
| `test_prune_checkpoints_keeps_recent`             | Checkpoint pruning keeps N most recent                       |
| `test_select_architecture_source`                 | Architecture source selection precedence                     |
| `test_select_*_checkpoint_path`                   | Various checkpoint path selection helpers                    |
| `test_should_save_checkpoint`                     | Checkpoint save decision at exact step boundaries            |
| `test_strip_compile_prefix`                       | `torch.compile` prefixes removed from state dict             |
| `test_coerce_rng_state`                           | RNG state coerced to uint8 CPU tensor                        |
| `test_apply_checkpoint_architecture_args_*`       | Checkpoint args restore model architecture                   |
| `test_normalize_training_accuracy`                | Training accuracy normalization helper                       |
| `test_accuracy_ignores_pad_tokens`                | Accuracy calculation ignores padding                         |
| `test_count_pred_tokens`                          | Predicted token counting with padding                        |
| `test_autocast_context_cpu_is_noop`               | CPU autocast is a no-op                                      |
| `test_ga_gradients_match_full_batch`              | Gradient accumulation matches full-batch gradients           |
| `test_scheduler_state_dict_is_stateless`          | Scheduler state_dict is stateless                            |

### `tests/test_pretrain_smoke.py` - End-to-End Smoke Tests

Integration tests that run actual training. **Requires CUDA.**

| Test                    | What It Verifies                                 |
| ----------------------- | ------------------------------------------------ |
| `test_resume_smoke_run` | Full train + resume cycle on GPU with tiny model |

This test:

1. Creates temporary train/valid/test files
2. Runs 1 training step with a minimal model
3. Verifies checkpoints are saved
4. Resumes from the checkpoint
5. Verifies resume completes without error

### `tests/dummy_tokenizer.py` - Test Fixture

A minimal tokenizer stub for offline unit tests. Implements the subset of the HuggingFace tokenizer API used by the codebase:

- `__call__` for encoding text
- `pad` for batch padding
- Special token IDs (PAD, CLS, SEP, MASK, UNK)
- `vocab_size` and `get_vocab()`

## Test Requirements

- **CPU tests** (`test_data.py`, `test_pretrain.py`): Run anywhere
- **GPU tests** (`test_pretrain_smoke.py`): Require CUDA; skipped automatically if unavailable

## Linting and Formatting

Always lint and format before committing:

```bash
ruff check --fix . && ruff format .
```

## Docstring Checks

Run the docstring checker against project code (avoid scanning vendored deps):

```bash
python ~/scripts/py/doc_check.py annotated_mpnet --check-lazy-docstrings
python ~/scripts/py/doc_check.py cli_tools --check-lazy-docstrings
```

## Development Workflow

1. Create a topic branch for your changes
2. Write tests for new functionality
3. Run the full test suite
4. Lint and format
5. Commit with descriptive messages

## Adding New Tests

Place new tests in the appropriate file based on what they test:

- Data loading/collation: `test_data.py`
- Model/training logic: `test_pretrain.py`
- End-to-end training flows: `test_pretrain_smoke.py`

Use `DummyTokenizer` for tests that don't need a real tokenizer to avoid network dependencies.
