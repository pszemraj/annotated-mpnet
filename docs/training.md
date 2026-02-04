# Training Guide

This document covers pretraining MPNet models using `annotated-mpnet`.

## Pretraining MPNet

The primary script for pretraining is `pretrain-mpnet`. You can see all available arguments by running `pretrain-mpnet -h`.

Training is **step-based** (no user-facing epochs). Datasets are cycled and reshuffled internally as needed, and training runs for `--total-updates` steps. Training metrics are logged every 25 update steps by default; adjust with `--logging-steps`.

### Using a HuggingFace Dataset (Streaming)

This method streams data directly from the HuggingFace Hub. Validation and test sets are created by taking initial samples from the training stream.

```bash
pretrain-mpnet \
    --dataset-name "gair-prox/DCLM-pro" \
    --text-field "text" \
    --tokenizer-name "microsoft/mpnet-base" \
    --max-tokens 512 \
    --encoder-layers 12 \
    --encoder-embed-dim 768 \
    --encoder-ffn-dim 3072 \
    --encoder-attention-heads 12 \
    --batch-size 16 \
    --update-freq 8 \
    --lr 6e-4 \
    --warmup-updates 1000 \
    --total-updates 100000 \
    --checkpoint-dir "./checkpoints/my_mpnet_run" \
    --tensorboard-log-dir "./logs/my_mpnet_run" \
    --wandb --wandb-project "annotated-mpnet-experiments" \
    --checkpoint-interval 2500 \
    --eval-interval-steps 2500
```

Key arguments for streaming:

- `--dataset-name`: Name of the dataset on HuggingFace Hub.
- `--text-field`: The column in the dataset containing the text (default: "text").
- `--buffer-size`: Size of the shuffling buffer for streaming (default: 10000).
- `--eval-samples`: Number of samples to take for validation/test sets from the stream (default: 500).
- `--min-text-length`: Minimum length of text samples to consider (default: 64).

### Using Local Text Files

Provide a directory of training files (one document/sentence per line is typical) and paths to single validation and test files.

```bash
pretrain-mpnet \
    --train-dir "/path/to/your/train_data_directory/" \
    --valid-file "/path/to/your/validation_data.txt" \
    --test-file "/path/to/your/test_data.txt" \
    --tokenizer-name "microsoft/mpnet-base" \
    --max-tokens 512 \
    --batch-size 16 \
    --update-freq 8 \
    --lr 6e-4 \
    --warmup-updates 1000 \
    --total-updates 100000 \
    --checkpoint-dir "./checkpoints/my_local_mpnet_run" \
    --tensorboard-log-dir "./logs/my_local_mpnet_run" \
    --checkpoint-interval 2500 \
    --eval-interval-steps 2500
```

## Key Pretraining Arguments

**Tokenizer and Sequence Length:**

- `--tokenizer-name`: HuggingFace tokenizer to use (default: `microsoft/mpnet-base`).
- `--max-tokens`: Maximum sequence length (default: 512). Also sets `--max-positions` if not specified.

**Model Architecture:**

- `--encoder-layers` (default: 12)
- `--encoder-embed-dim` (default: 768)
- `--encoder-ffn-dim` (default: 3072)
- `--encoder-attention-heads` (default: 12)

**Training Parameters:**

- `--batch-size`: Per-GPU batch size (default: 16).
- `--update-freq`: Gradient accumulation steps to simulate larger batch sizes (default: 8). Effective batch size = `batch-size * update-freq * num_gpus`.
- `--gradient-checkpointing`: Enable activation checkpointing to reduce memory usage (adds recompute).
- Mixed precision: CUDA runs use bf16 autocast by default. FP16 is not wired and would require adding GradScaler support.
- `--lr`: Peak learning rate (default: 6e-4).
- `--warmup-updates`: Number of steps for LR warmup (default: 10% of `total-updates`).
- `--total-updates`: Total number of training updates (default: 10000).

**Logging and Saving:**

- `--checkpoint-dir`: Directory to save model checkpoints (default: `./checkpoints`).
- `--tensorboard-log-dir`: Directory for TensorBoard logs. If unset, logs to console.
- `--checkpoint-interval`: Save a checkpoint every N steps (default: -1, only best and final). Alias: `--save_steps`.
- `--keep-checkpoints`: Keep the most recent N interval checkpoints (-1 disables pruning; 0 keeps none).
- `--eval-interval-steps`: Run validation every N steps (default: `--checkpoint-interval` if set, otherwise 5000).
- `--wandb`: Enable Weights & Biases logging.
- `--wandb-project`, `--wandb-name`: W&B project and run name.

**Other:**

- `--compile`: Use `torch.compile()` for the model (experimental, default: False).
- `--seed`: Random seed for reproducibility (default: 12345).

**Data Source Selection:**

- If `--dataset-name` is omitted, the script defaults to the streaming dataset `HuggingFaceFW/fineweb-edu`.
- If you provide `--train-dir`, `--valid-file`, and `--test-file`, the file-based path is used automatically (no need to pass `--dataset-name ""`).

The script validates the tokenizer. For optimal performance with the default `whole_word_mask=True` in the data collator, a WordPiece-compatible tokenizer is expected.

## Resuming Training

Resuming is supported **only for checkpoints created by v0.1.5+** (they include `data_state` for step-based training). Legacy checkpoints from earlier versions are **not** resumable; they can only be used to initialize weights, and all optimizer/scheduler/step state will be reset.

To enable full resume, save optimizer state during training:

```bash
pretrain-mpnet \
    --dataset-name "gair-prox/DCLM-pro" \
    --tokenizer-name "microsoft/mpnet-base" \
    --total-updates 200000 \
    --checkpoint-dir "./checkpoints/my_mpnet_run" \
    --checkpoint-interval 2500 \
    --save-optimizer-state
```

Then resume:

```bash
pretrain-mpnet \
    --dataset-name "gair-prox/DCLM-pro" \
    --tokenizer-name "microsoft/mpnet-base" \
    --total-updates 200000 \
    --checkpoint-dir "./checkpoints/my_mpnet_run" \
    --resume \
    --resume-checkpoint "./checkpoints/my_mpnet_run/checkpoint2500.pt"
```

**Notes:**

- If you pass `--resume` with a legacy checkpoint (pre-v0.1.5), the script will **only load weights** and start fresh from step 0.
- To initialize from a legacy checkpoint, you can also convert it to HuggingFace format and pass `--hf-model-path`.
- `--hf-model-path` cannot be combined with `--resume` or `--resume-checkpoint`.
- Checkpoints are loaded with safe `weights_only` by default. Use `--trust-checkpoint` to load legacy or external `.pt` files.
- Resuming requires a tokenizer with the same vocab size as the checkpoint/HF config. Mismatches will raise an error.
- Streaming datasets rely on HuggingFace's `shuffle()` per cycle; resume uses cycle + sample offset (no extra dataset-level shuffling).
- For streaming datasets, deterministic resume requires `--num-workers 0`; with more workers, data order is best-effort via sample skipping.
- If `--resume` is set without `--resume-checkpoint`, the latest interval checkpoint is used; if none exist, it falls back to `best_checkpoint.pt`.
- If validation/test datasets are empty or disabled, validation/test evaluation is skipped and the final test eval falls back to the in-memory model.

## Exporting Checkpoint to HuggingFace

After pretraining, convert your checkpoint to the HuggingFace `MPNetForMaskedLM` format using the `convert-to-hf` script. This allows you to load and use your model within the HuggingFace ecosystem.

```bash
convert-to-hf \
    --mpnet-checkpoint-path "./checkpoints/my_mpnet_run/best_checkpoint.pt" \
    --hf-model-folder-path "./my_hf_mpnet_model/"
```

- By default, this script will also save the tokenizer used during pretraining (if its name was stored in the checkpoint args). Use `--no-save-tokenizer` to disable this.
- The output directory (`./my_hf_mpnet_model/`) will contain `pytorch_model.bin`, `config.json`, and tokenizer files (e.g., `tokenizer.json`, `vocab.txt`).
