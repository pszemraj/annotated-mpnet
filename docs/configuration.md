# Configuration Reference

Complete reference for all `pretrain-mpnet` arguments. Default values follow the MPNet-base parameters from [the paper](https://arxiv.org/abs/2004.09297).

---

- [Model Architecture](#model-architecture)
  - [Activation Functions](#activation-functions)
- [Positional Encoding](#positional-encoding)
- [Tokenizer](#tokenizer)
- [Data Source](#data-source)
  - [HuggingFace Streaming (default)](#huggingface-streaming-default)
  - [Local Files](#local-files)
- [Training](#training)
- [Learning Rate Schedule](#learning-rate-schedule)
- [Optimizer (AdamW)](#optimizer-adamw)
- [Checkpoints](#checkpoints)
- [Resume Training](#resume-training)
- [Evaluation](#evaluation)
- [Logging](#logging)
- [Weights \& Biases](#weights--biases)
- [Mutual Exclusivity and Constraints](#mutual-exclusivity-and-constraints)
- [Quick Reference: Common Configurations](#quick-reference-common-configurations)
  - [MPNet-base (default)](#mpnet-base-default)
  - [Memory-constrained training](#memory-constrained-training)
  - [Resumable training](#resumable-training)
  - [Full logging](#full-logging)

---

## Model Architecture

| Argument                         | Type    | Default | Description                                                                                                                                        |
| -------------------------------- | ------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--encoder-layers`               | `int`   | `12`    | Number of transformer encoder layers. MPNet-base uses 12; larger models typically use 12--24.                                                      |
| `--encoder-embed-dim`            | `int`   | `768`   | Dimension of token embeddings and hidden states. Must be divisible by `--encoder-attention-heads`. Common values: 768 (base), 1024 (large).        |
| `--encoder-ffn-dim`              | `int`   | `3072`  | Dimension of the feed-forward network hidden layer. Typically 3--4× `--encoder-embed-dim`.                                                         |
| `--encoder-attention-heads`      | `int`   | `12`    | Number of attention heads per layer. `--encoder-embed-dim` must be divisible by this value.                                                        |
| `--dropout`                      | `float` | `0.1`   | Dropout probability applied throughout the encoder (embeddings, attention output, FFN output).                                                     |
| `--attention-dropout`            | `float` | `0.1`   | Dropout probability applied to attention weights after softmax.                                                                                    |
| `--activation-dropout`           | `float` | `0.1`   | Dropout probability applied after the activation function in the FFN hidden layer.                                                                 |
| `--normalize-before`, `-prenorm` | `bool`  | `False` | Apply layer normalization before sublayer operations (Pre-LN) instead of after (Post-LN). Pre-LN can improve training stability for deeper models. |

### Activation Functions

The `--activation-fn` (alias `-activation`) argument sets the FFN activation function. Type: `str`, default: `"gelu"`.

| Value           | Description                              |
| --------------- | ---------------------------------------- |
| `gelu`          | Gaussian Error Linear Unit (recommended) |
| `gelu_accurate` | More accurate GELU approximation         |
| `relu`          | Rectified Linear Unit                    |
| `relu2`         | Squared ReLU                             |
| `silu`          | Sigmoid Linear Unit (SwiGLU)             |
| `tanh`          | Hyperbolic tangent                       |
| `linear`        | No activation (identity)                 |

---

## Positional Encoding

| Argument                                           | Type  | Default | Description                                                                                                                     |
| -------------------------------------------------- | ----- | ------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `--max-tokens`                                     | `int` | `512`   | Maximum sequence length for input tokens. Sequences longer than this are truncated. Also sets `--max-positions` if unspecified. |
| `--max-positions`                                  | `int` | `None`  | Maximum number of positional embeddings. Defaults to `--max-tokens` if unset.                                                   |
| `--relative-attention-num-buckets`, `-num_buckets` | `int` | `None`  | Number of buckets for relative position bias. If unset, automatically computed from sequence length. Typical value: 32.         |
| `--relative-attention-max-distance`                | `int` | `None`  | Maximum distance for relative position encoding. If unset, automatically computed from sequence length. Typical value: 128.     |

> [!NOTE]
> `--max-tokens` and `--max-positions` should almost always be the same[^1], so only set one of them.

[^1]: An exception would be if you plan to do multiple training phases and gradually increase the context length trained at (_for efficiency reasons_). In this case, `--max-positions` should be set to the **final planned context length** (e.g. 2048) and `--max-tokens` should be set to the **initial training phase context length** (e.g. 512)

---

## Tokenizer

> [!IMPORTANT]
> For optimal whole-word masking (`whole_word_mask=True` in the data collator), use a WordPiece-compatible tokenizer.

| Argument           | Type  | Default                  | Description                                                                                                                              |
| ------------------ | ----- | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `--tokenizer-name` | `str` | `"microsoft/mpnet-base"` | HuggingFace tokenizer name or path to local tokenizer directory. The tokenizer's vocabulary determines the model's embedding layer size. |

---

## Data Source

You must use **either** HuggingFace streaming **or** local files, not both.

### HuggingFace Streaming (default)

| Argument            | Type  | Default  | Description                                                                                                                                                                                                    |
| ------------------- | ----- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--dataset-name`    | `str` | `None`   | HuggingFace dataset name (e.g., `"HuggingFaceFW/fineweb-edu"`, `"gair-prox/DCLM-pro"`). When set, overrides file-based paths. Defaults to `"HuggingFaceFW/fineweb-edu"` when no file-based paths are provided. |
| `--text-field`      | `str` | `"text"` | Column name in the dataset containing the text to tokenize.                                                                                                                                                    |
| `--buffer-size`     | `int` | `10000`  | Size of the shuffle buffer for streaming datasets. Larger buffers give better randomization but use more memory. Recommended: 10000--100000.                                                                   |
| `--eval-samples`    | `int` | `500`    | Number of samples to reserve for validation and test sets when streaming. These are taken from the start of the stream.                                                                                        |
| `--min-text-length` | `int` | `64`     | Minimum character length for text samples. Samples shorter than this are skipped. Set to 0 to disable filtering.                                                                                               |

### Local Files

| Argument       | Type  | Default | Description                                                                           |
| -------------- | ----- | ------- | ------------------------------------------------------------------------------------- |
| `--train-dir`  | `str` | `None`  | Directory containing training files. Each file is fully loaded into memory per cycle. |
| `--valid-file` | `str` | `None`  | Path to validation data file. One document/sentence per line.                         |
| `--test-file`  | `str` | `None`  | Path to test data file. One document/sentence per line.                               |

> [!WARNING]
> For large corpora, prefer `--dataset-name` streaming to avoid running out of CPU RAM.

---

## Training

| Argument                     | Type   | Default | Description                                                                                                                               |
| ---------------------------- | ------ | ------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `--total-updates`            | `int`  | `10000` | Total number of optimizer update steps. Training is step-based, not epoch-based. The dataset cycles internally as needed.                 |
| `--batch-size`               | `int`  | `16`    | Per-GPU batch size (number of sequences per forward pass).                                                                                |
| `--update-freq`, `-gc_steps` | `int`  | `8`     | Gradient accumulation steps. Effective batch size = `batch-size × update-freq × num_gpus`.                                                |
| `--gradient-checkpointing`   | `bool` | `False` | Enable activation checkpointing to reduce memory usage at the cost of ~20% slower training. Recommended for large models or limited VRAM. |
| `--compile`                  | `bool` | `False` | Use `torch.compile()` for potential speedup. Experimental; may not work with all configurations.                                          |
| `--seed`                     | `int`  | `12345` | Random seed for reproducibility. Affects model initialization, data shuffling, and masking.                                               |
| `--num-workers`              | `int`  | `0`     | Number of DataLoader worker processes.                                                                                                    |

> [!IMPORTANT]
> Deterministic resume requires `--num-workers 0`. With more workers, data order is best-effort via sample skipping.

**Example:** `--batch-size 16 --update-freq 8` on 1 GPU = effective batch size of 128.

---

## Learning Rate Schedule

Uses polynomial decay with linear warmup.

| Argument                            | Type    | Default | Description                                                                                        |
| ----------------------------------- | ------- | ------- | -------------------------------------------------------------------------------------------------- |
| `--lr`                              | `float` | `6e-4`  | Peak learning rate reached after warmup completes.                                                 |
| `--warmup-updates`, `-warmup_steps` | `int`   | `None`  | Number of warmup steps with linearly increasing LR. Defaults to 10% of `--total-updates` if unset. |
| `--end-learning-rate`, `-end_lr`    | `float` | `0.0`   | Final learning rate after polynomial decay. Set to 0 for full decay.                               |
| `--power`                           | `float` | `1.0`   | Power of the polynomial decay. `1.0` = linear decay; `2.0` = quadratic decay.                      |

---

## Optimizer (AdamW)

| Argument                         | Type    | Default | Description                                                                               |
| -------------------------------- | ------- | ------- | ----------------------------------------------------------------------------------------- |
| `--beta1`                        | `float` | `0.9`   | AdamW first moment coefficient.                                                           |
| `--beta2`                        | `float` | `0.98`  | AdamW second moment coefficient. MPNet uses 0.98 (higher than typical 0.999).             |
| `--weight-decay`, `-wd`          | `float` | `0.01`  | Weight decay coefficient. Applied to all parameters except biases and layer norm weights. |
| `--adam-eps`                     | `float` | `1e-6`  | Epsilon for numerical stability in AdamW.                                                 |
| `--clip-grad-norm`, `-grad_clip` | `float` | `1.0`   | Maximum gradient norm for gradient clipping. Set to 0 to disable.                         |

---

## Checkpoints

| Argument                                | Type   | Default           | Description                                                                                                           |
| --------------------------------------- | ------ | ----------------- | --------------------------------------------------------------------------------------------------------------------- |
| `--checkpoint-dir`                      | `str`  | `"./checkpoints"` | Directory for saving checkpoints. Creates `best_checkpoint.pt`, `final_checkpoint.pt`, and interval checkpoints.      |
| `--checkpoint-interval`, `--save-steps` | `int`  | `-1`              | Save an interval checkpoint every N update steps. `-1` disables interval checkpoints (only best and final are saved). |
| `--keep-checkpoints`                    | `int`  | `-1`              | Number of interval checkpoints to keep. `-1` = keep all, `0` = keep none (only best/final), `N` = keep N most recent. |
| `--save-optimizer-state`                | `bool` | `False`           | Save optimizer and scheduler state alongside model weights. **Required for full resume.**                             |

---

## Resume Training

| Argument              | Type   | Default | Description                                                                                                                                   |
| --------------------- | ------ | ------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `--resume`            | `bool` | `False` | Enable resume mode. Loads model, optimizer, scheduler, and data state from checkpoint.                                                        |
| `--resume-checkpoint` | `str`  | `None`  | Explicit path to checkpoint file for resuming. If unset with `--resume`: uses latest interval checkpoint, falls back to `best_checkpoint.pt`. |
| `--trust-checkpoint`  | `bool` | `False` | Allow loading checkpoints without `weights_only=True`. Required for legacy or external `.pt` files.                                           |
| `--hf-model-path`     | `str`  | `None`  | Path to HuggingFace MPNet model to initialize weights from. Alternative to resuming from repo checkpoints.                                    |

> [!NOTE]
> Full resume requires checkpoints created by v0.1.5+ (with `data_state`). Legacy checkpoints only initialize weights.

---

## Evaluation

| Argument                | Type  | Default | Description                                                                                      |
| ----------------------- | ----- | ------- | ------------------------------------------------------------------------------------------------ |
| `--eval-interval-steps` | `int` | `None`  | Run validation every N update steps. Defaults to `--checkpoint-interval` if set, otherwise 5000. |

---

## Logging

| Argument                            | Type   | Default | Description                                                                   |
| ----------------------------------- | ------ | ------- | ----------------------------------------------------------------------------- |
| `--tensorboard-log-dir`, `-log_dir` | `str`  | `None`  | Directory for TensorBoard logs. If unset, metrics are logged to console only. |
| `--logging-steps`                   | `int`  | `25`    | Log training metrics every N update steps.                                    |
| `--debug`                           | `bool` | `False` | Enable debug logging (verbose output).                                        |

---

## Weights & Biases

| Argument          | Type   | Default             | Description                                                                                     |
| ----------------- | ------ | ------------------- | ----------------------------------------------------------------------------------------------- |
| `--wandb`         | `bool` | `False`             | Enable Weights & Biases logging.                                                                |
| `--wandb-project` | `str`  | `"annotated-mpnet"` | W&B project name.                                                                               |
| `--wandb-name`    | `str`  | `None`              | W&B run name. Auto-generated if unset.                                                          |
| `--wandb-id`      | `str`  | `None`              | W&B run ID for resuming a tracked run. Use with `--resume` to continue logging to the same run. |
| `--wandb-watch`   | `bool` | `False`             | Log model gradients to W&B. Increases logging overhead.                                         |

---

## Mutual Exclusivity and Constraints

| Constraint                          | Details                                                   |
| ----------------------------------- | --------------------------------------------------------- |
| `--dataset-name` vs `--train-dir`   | Cannot use both; streaming overrides file-based           |
| `--hf-model-path` vs `--resume`     | Cannot combine; use one or the other                      |
| `--max-tokens` vs `--max-positions` | Set only one; they must match if both are set             |
| Deterministic resume                | Requires `--num-workers 0` for streaming datasets         |
| Full resume                         | Requires `--save-optimizer-state` during initial training |
| Full resume                         | Requires checkpoint from v0.1.5+ (with `data_state`)      |

---

## Quick Reference: Common Configurations

> [!NOTE]
> The default streaming dataset is [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)

### MPNet-base (default)

```bash
pretrain-mpnet \
    --encoder-layers 12 \
    --encoder-embed-dim 768 \
    --encoder-ffn-dim 3072 \
    --encoder-attention-heads 12
```

### Memory-constrained training

```bash
pretrain-mpnet \
    --batch-size 8 \
    --update-freq 16 \
    --gradient-checkpointing
```

### Resumable training

```bash
# Initial run
pretrain-mpnet \
    --save-optimizer-state \
    --checkpoint-interval 2500

# Resume
pretrain-mpnet \
    --resume \
    --resume-checkpoint ./checkpoints/checkpoint2500.pt
```

### Full logging

```bash
pretrain-mpnet \
    --tensorboard-log-dir ./logs \
    --wandb \
    --wandb-project my-project \
    --logging-steps 10
```
