# Annotated MPNet

`annotated-mpnet` provides a lightweight, heavily annotated, and PyTorch/einops implementation for pretraining MPNet models. This project aims to demystify the MPNet pretraining process, originally part of the larger `fairseq` codebase, making it more accessible for research and custom pretraining.

---

- [About the Project](#about-the-project)
- [Key Features](#key-features)
- [Installation](#installation)
  - [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Project Structure](#project-structure)
- [Changelog](#changelog)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## About the Project

> [!NOTE]
> **This repo** is a fork/update of the [original by yext](https://github.com/yext/annotated-mpnet).

MPNet ([Masked and Permuted Pre-training for Language Understanding](https://arxiv.org/abs/2004.09297)) is a powerful pretraining method. However, its [original pretraining code](https://github.com/microsoft/MPNet) is embedded within the `fairseq` library, which can be complex to navigate and adapt. `annotated-mpnet` addresses this by:

- Providing a clean, raw PyTorch implementation of MPNet pretraining.
- Offering extensive annotations and comments throughout the codebase to improve understanding.
- Enabling pretraining without the full `fairseq` dependency, facilitating use on various hardware setups.

## Key Features

- **Standalone PyTorch Implementation**: No `fairseq` dependency required for pretraining.
- **Heavily Annotated Code**: Detailed comments explain the model architecture and training process.
- **Flexible Data Handling**: Supports pretraining with HuggingFace streaming datasets or local text files.
- **Optional RoPE + FlexAttention Path**: Supports rotary embeddings and structural mask-based attention with SDPA fallback.
- **HuggingFace Compatibility**: Includes a tool to convert pretrained checkpoints to the HuggingFace `MPNetForMaskedLM` format for easy fine-tuning.
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
- `einops` (version >= 0.7.0) for explicit tensor shape transforms
- CUDA GPU with BF16 support (Ampere+). As of 2026, legacy GPUs without BF16 are not supported.
- HuggingFace `transformers`, `datasets`
- `wandb` (for Weights & Biases logging, optional)
- `rich` (for enhanced console logging)
- `numpy`, `cython`, `tensorboard` (optional)

See `pyproject.toml` for a full list of dependencies.

## Quick Start

Stream data from HuggingFace and start pretraining:

```bash
pretrain-mpnet \
    --dataset-name "HuggingFaceFW/fineweb-edu" \
    --tokenizer-name "microsoft/mpnet-base" \
    --batch-size 16 \
    --update-freq 8 \
    --total-updates 100000 \
    --checkpoint-dir "./checkpoints/my_run"
```

Run `pretrain-mpnet -h` for all available options.

Optional RoPE + FlexAttention run (with explicit backend override):

```bash
pretrain-mpnet \
    --dataset-name "HuggingFaceFW/fineweb-edu" \
    --tokenizer-name "microsoft/mpnet-base" \
    --batch-size 16 \
    --total-updates 100000 \
    --use-rope \
    --no-relative-attention-bias \
    --attention-dropout 0.0 \
    --use-flex-attention \
    --flex-backend triton \
    --checkpoint-dir "./checkpoints/my_rope_flex_run"
```

## Documentation

- **[Training Guide](docs/training.md)** - Full usage: streaming vs. local data, resuming, exporting to HuggingFace
- **[Configuration Reference](docs/configuration.md)** - Complete reference for all CLI arguments
- **[Architecture](docs/architecture.md)** - Model internals: two-stream attention, encoder structure
- **[Development Guide](docs/dev.md)** - Running tests, contributing

## Project Structure

```text
annotated-mpnet/
├── annotated_mpnet/          # Core library code
│   ├── data/                 # Data loading, collation, streaming dataset
│   ├── modeling/             # MPNetForPretraining model definition
│   ├── scheduler/            # Learning rate scheduler
│   ├── tracking/             # Metrics tracking (AverageMeter)
│   ├── transformer_modules/  # Transformer building blocks
│   └── utils/                # Utilities, Cython-accelerated permutation
├── cli_tools/                # CLI scripts (pretrain-mpnet, convert-to-hf)
├── docs/                     # Documentation
├── tests/                    # Unit tests
└── pyproject.toml            # Build configuration
```

## Changelog

All notable changes to this project are documented in [CHANGELOG.md](CHANGELOG.md). The latest version is v0.1.6.

## Contributing

Contributions are welcome! Please consider the following:

- **Reporting Issues**: Use GitHub Issues to report bugs or suggest new features.
- **Pull Requests**: For code contributions, please open a pull request with a clear description of your changes.
- **Running Tests**: Ensure tests pass with `python -m unittest discover tests`.

## License

The licenses for third-party libraries used in this project are detailed in [LICENSE-3RD-PARTY.txt](LICENSE-3RD-PARTY.txt). The original MPNet code by Microsoft is licensed under the MIT License.

> [!NOTE]
> The detailed line-by-line license info is from the original repo and has not been updated in this fork.

## Acknowledgements

- This work is heavily based on the original MPNet paper and implementation by Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu from Microsoft.
- The core Transformer module structures are adapted from the `fairseq` library.
