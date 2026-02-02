# Annotated MPNet

`annotated-mpnet` provides a lightweight, heavily annotated, and standalone PyTorch implementation for pretraining MPNet models. This project aims to demystify the MPNet pretraining process, which was originally part of the larger `fairseq` codebase, making it more accessible for research and custom pretraining.

## Table of Contents

- [Annotated MPNet](#annotated-mpnet)
  - [Table of Contents](#table-of-contents)
  - [About the Project](#about-the-project)
  - [Key Features](#key-features)
  - [Installation](#installation)
    - [Requirements](#requirements)
  - [Usage](#usage)
    - [Pretraining MPNet](#pretraining-mpnet)
    - [Resuming Training](#resuming-training)
    - [Porting Checkpoint to Hugging Face](#porting-checkpoint-to-hugging-face)
  - [Model Architecture](#model-architecture)
  - [Project Structure](#project-structure)
  - [Changelog](#changelog)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

## About the Project

MPNet (Masked and Permuted Pre-training for Language Understanding) is a powerful pretraining method. However, its original pretraining code is embedded within the `fairseq` library, which can be complex to navigate and adapt. `annotated-mpnet` addresses this by:

- Providing a clean, raw PyTorch implementation of MPNet pretraining.
- Offering extensive annotations and comments throughout the codebase to improve understanding.
- Enabling pretraining without the full `fairseq` dependency, facilitating use on various hardware setups.

> [!NOTE]
> **This repo** is a fork/update of the [original by yext](https://github.com/yext/annotated-mpnet).

## Key Features

- **Standalone PyTorch Implementation**: No `fairseq` dependency required for pretraining.
- **Heavily Annotated Code**: Detailed comments explain the model architecture and training process.
- **Flexible Data Handling**: Supports pretraining with Hugging Face streaming datasets or local text files.
- **Hugging Face Compatibility**: Includes a tool to convert pretrained checkpoints to the Hugging Face `MPNetForMaskedLM` format for easy fine-tuning.
- **Integrated Logging**: Supports TensorBoard and Weights & Biases for experiment tracking.

## Installation

pip install directly from the GitHub repository:

```bash
pip install "git+https://github.com/pszemraj/annotated-mpnet.git"
```

Or, clone the repository and install in editable mode:

```bash
git clone https://github.com/pszemraj/annotated-mpnet.git
cd annotated-mpnet
pip install -e .
```

> [!NOTE]
> Pretraining MPNet is computationally intensive and requires a CUDA-enabled GPU. The training script will exit if CUDA is not available.

### Requirements

- Python 3.x
- PyTorch (version >= 2.6.0, CUDA is required for training)
- Hugging Face `transformers`, `datasets`
- `wandb` (for Weights & Biases logging, optional)
- `rich` (for enhanced console logging)
- `numpy`
- `cython`
- `tensorboard` (for logging, optional)

See `setup.py` for a full list of dependencies.

## Usage

### Pretraining MPNet

The primary script for pretraining is `pretrain-mpnet`. You can see all available arguments by running `pretrain-mpnet -h`.
Training is **step-based** (no user-facing epochs). Datasets are cycled and reshuffled internally as needed, and training runs for `--total-updates` steps.

**1. Using a Hugging Face Dataset (Streaming):**
This method streams data directly from the Hugging Face Hub. Validation and test sets are created by taking initial samples from the training stream.

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

- `--dataset-name`: Name of the dataset on Hugging Face Hub.
- `--text-field`: The column in the dataset containing the text (default: "text").
- `--buffer-size`: Size of the shuffling buffer for streaming (default: 10000).
- `--eval-samples`: Number of samples to take for validation/test sets from the stream (default: 500).
- `--min-text-length`: Minimum length of text samples to consider (default: 64).

**2. Using Local Text Files:**
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

**Key Pretraining Arguments (Common to both methods):**

- `--tokenizer-name`: Hugging Face tokenizer to use (default: `microsoft/mpnet-base`).
- `--max-tokens`: Maximum sequence length (default: 512). Also sets `--max-positions` if not specified.
- Model Architecture:
  - `--encoder-layers` (default: 12)
  - `--encoder-embed-dim` (default: 768)
  - `--encoder-ffn-dim` (default: 3072)
  - `--encoder-attention-heads` (default: 12)
- Training Parameters:
  - `--batch-size`: Per-GPU batch size (default: 16).
  - `--update-freq`: Gradient accumulation steps to simulate larger batch sizes (default: 8). Effective batch size = `batch-size * update-freq * num_gpus`.
  - `--gradient-checkpointing`: Enable activation checkpointing to reduce memory usage (adds recompute).
  - Mixed precision: CUDA runs use bf16 autocast by default. FP16 is not wired and would require adding GradScaler support.
  - `--lr`: Peak learning rate (default: 6e-4).
  - `--warmup-updates`: Number of steps for LR warmup (default: 10% of `total-updates`).
  - `--total-updates`: Total number of training updates (default: 10000).
- Logging and Saving:
  - `--checkpoint-dir`: Directory to save model checkpoints (default: `./checkpoints`).
  - `--tensorboard-log-dir`: Directory for TensorBoard logs. If unset, logs to console.
  - `--checkpoint-interval`: Save a checkpoint every N steps (default: -1, only best and final). Alias: `--save_steps`.
  - `--keep-checkpoints`: Keep the most recent N interval checkpoints (-1 disables pruning; 0 keeps none).
  - `--eval-interval-steps`: Run validation every N steps (default: `--checkpoint-interval` if set, otherwise 5000).
  - `--wandb`: Enable Weights & Biases logging.
  - `--wandb-project`, `--wandb-name`: W\&B project and run name.
- `--compile`: Use `torch.compile()` for the model (experimental, default: False).
- `--seed`: Random seed for reproducibility (default: 12345).

Data source selection:

- If `--dataset-name` is omitted, the script defaults to the streaming dataset `HuggingFaceFW/fineweb-edu`.
- If you provide `--train-dir`, `--valid-file`, and `--test-file`, the file-based path is used automatically (no need to pass `--dataset-name ""`).

The script validates the tokenizer. For optimal performance with the default `whole_word_mask=True` in the data collator, a WordPiece-compatible tokenizer is expected.

### Resuming Training

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

Notes:

- If you pass `--resume` with a legacy checkpoint (pre-v0.1.5), the script will **only load weights** and start fresh from step 0.
- To initialize from a legacy checkpoint, you can also convert it to Hugging Face format and pass `--hf-model-path`.
- Checkpoints are loaded with safe `weights_only` by default. Use `--trust-checkpoint` to load legacy or external `.pt` files.

### Porting Checkpoint to Hugging Face

After pretraining, convert your checkpoint to the Hugging Face `MPNetForMaskedLM` format using the `convert-to-hf` script. This allows you to load and use your model within the Hugging Face ecosystem.

```bash
convert-to-hf \
    --mpnet-checkpoint-path "./checkpoints/my_mpnet_run/best_checkpoint.pt" \
    --hf-model-folder-path "./my_hf_mpnet_model/"
```

- By default, this script will also save the tokenizer used during pretraining (if its name was stored in the checkpoint args). Use `--no-save-tokenizer` to disable this.
- The output directory (`./my_hf_mpnet_model/`) will contain `pytorch_model.bin`, `config.json`, and tokenizer files (e.g., `tokenizer.json`, `vocab.txt`).

## Model Architecture

This repository implements MPNet, which utilizes a **Masked and Permuted Pre-training** objective. The architecture is based on the Transformer model.

- **`MPNetForPretraining`**: This is the main model class defined in `annotated_mpnet/modeling/mpnet_for_pretraining.py`. It encapsulates the encoder and the language modeling head.
- **`SentenceEncoder`**: The core of the model, this is a stack of Transformer encoder layers. It's responsible for generating contextualized representations of the input tokens. Found in `annotated_mpnet/transformer_modules/sentence_encoder.py`.
- **`SentenceEncoderLayer`**: Each layer within the `SentenceEncoder`. It primarily consists of:
  - **`RelativeMultiHeadAttention`**: A multi-head self-attention mechanism that incorporates relative positional information, crucial for MPNet. Defined in `annotated_mpnet/transformer_modules/rel_multihead_attention.py`.
  - Position-wise Feed-Forward Networks (FFN).
  - Layer normalization.
- **Positional Embeddings**: The model uses positional embeddings to provide sequence order information. This implementation supports:
  - `LearnedPositionalEmbedding`: Positional embeddings are learned during training.
  - `SinusoidalPositionalEmbedding`: Fixed positional embeddings based on sine and cosine functions.
        The choice is configurable via `pretrain_mpnet.py` arguments. These are found in `annotated_mpnet/transformer_modules/`.
- **Two-Stream Self-Attention**: A key innovation of MPNet. While not a separate module, this mechanism is implemented within the `MPNetForPretraining` forward pass. It allows the model to predict original tokens from a permuted version of the input by using two streams of information (content and query), enabling it to learn bidirectional context without the predicted tokens "seeing themselves" in the non-permuted context.
- **`MPNetLMHead`**: A language modeling head placed on top of the `SentenceEncoder`'s output. It projects the contextual embeddings to the vocabulary space to predict the masked tokens. Defined in `annotated_mpnet/modeling/mpnet_for_pretraining.py`.
- **Normalization Strategy**: The `--normalize-before` flag (default: `False` in `SentenceEncoder`, `True` for `encoder_normalize_before` in `MPNetForPretraining`) controls whether layer normalization is applied before or after sublayer operations (attention and FFN), following common Transformer variations.

The pretraining objective involves predicting original tokens based on a permuted sequence where a subset of tokens has been masked. The permutation helps in learning richer contextual representations compared to standard Masked Language Modeling (MLM).

## Project Structure

```text
annotated-mpnet/
├── annotated_mpnet/          # Core library code
│   ├── data/                 # Data loading, collation, (HF) streaming dataset
│   ├── modeling/             # MPNetForPretraining model definition
│   ├── scheduler/            # Learning rate scheduler
│   ├── tracking/             # Metrics tracking (AverageMeter)
│   ├── transformer_modules/  # Core Transformer building blocks (attention, layers, embeddings)
│   └── utils/                # Utility functions, including Cython-accelerated permutation
├── cli_tools/                # Command-line interface scripts
│   ├── pretrain_mpnet.py
│   └── convert_pretrained_mpnet_to_hf_model.py
├── tests/                    # Unit tests
├── checkpoints/              # Default directory for saved model checkpoints
├── LICENSE-3RD-PARTY.txt     # Licenses for third-party dependencies
├── README.md                 # This file
├── CHANGELOG.md              # Record of changes
└── setup.py                  # Package setup script
```

## Changelog

All notable changes to this project are documented in [CHANGELOG.md](CHANGELOG.md). The latest version is v0.1.6.

## Contributing

Contributions are welcome\! Please consider the following:

- **Reporting Issues**: Use GitHub Issues to report bugs or suggest new features.
- **Pull Requests**: For code contributions, please open a pull request with a clear description of your changes.
- **Running Tests**: Ensure tests pass. You can run tests using:

    ```bash
    python -m unittest discover tests
    ```

## License

The licenses for third-party libraries used in this project are detailed in [LICENSE-3RD-PARTY.txt](https://www.google.com/search?q=LICENSE-3RD-PARTY.txt). The original MPNet code by Microsoft is licensed under the MIT License. The specific licensing for contributions made within this `annotated-mpnet` repository should be determined by its maintainers; users should refer to any specific license file provided at the root of this repository or assume standard open-source licensing practices.

Note that the detailed line-by-line license info is from the original repo and has not been updated in this fork.

## Acknowledgements

- This work is heavily based on the original MPNet paper and implementation by Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu from Microsoft.
- The core Transformer module structures are adapted from the `fairseq` library.
