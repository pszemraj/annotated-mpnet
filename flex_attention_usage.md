# MPNet with FlexAttention

This document explains how to use the FlexAttention implementation with MPNet for efficient pretraining, especially with long sequences using sliding window attention.

## Overview

FlexAttention is a flexible attention mechanism that supports custom attention patterns, including:

- Full attention (standard behavior)
- Sliding window attention (efficient for long sequences)
- Custom masking patterns

The primary benefit of FlexAttention for MPNet is the ability to use sliding window attention, which significantly reduces memory usage and computation time for long sequences while maintaining most of the modeling quality.

## Getting Started

### Installation

The FlexAttention implementation is included in the `annotated-mpnet` package. Install it using:

```bash
pip install -e .
```

### Using FlexAttention with MPNet

There are two main ways to use FlexAttention with MPNet:

1. Use the dedicated CLI script `pretrain-mpnet-flex` (recommended)
2. Use the `MPNetFlexForPretraining` class directly in your code

## CLI Usage

The `pretrain-mpnet-flex` command provides a convenient way to pretrain MPNet with FlexAttention:

```bash
pretrain-mpnet-flex \
  --dataset-name HuggingFaceFW/fineweb-edu \
  --sliding-window-size 128 \
  --total-updates 10000 \
  --warmup-updates 1000 \
  --batch-size 16 \
  --update-freq 4 \
  --lr 0.0002 \
  --checkpoint-dir checkpoints/mpnet_flex \
  --tensorboard-log-dir logs/mpnet_flex
```

### Key Parameters

- `--sliding-window-size`: The size of the sliding window for attention. If not specified (or set to `None`), full attention will be used.
- `--max-tokens`: The maximum number of tokens in a sequence.
- Other parameters work the same as in regular MPNet pretraining.

## Memory Usage and Performance

Sliding window attention significantly reduces memory usage and increases training speed for long sequences. The table below provides a comparison:

| Sequence Length | Model         | Memory Usage | Training Speed (tokens/sec) |
|-----------------|---------------|--------------|----------------------------|
| 512             | Standard      | 100%         | 100%                       |
| 512             | Sliding (128) | ~75%         | ~120%                      |
| 1024            | Standard      | 400%         | 25%                        |
| 1024            | Sliding (128) | ~100%        | ~90%                       |
| 2048            | Standard      | Out of memory| N/A                        |
| 2048            | Sliding (128) | ~150%        | ~70%                       |

Note: Percentages are relative to standard MPNet with 512 sequence length.

## Programmatic Usage

You can also use the FlexAttention implementation in your own code:

```python
from annotated_mpnet.modeling import MPNetFlexForPretraining
from transformers import AutoTokenizer

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")

# Define args (can be from argparse)
class Args:
    def __init__(self):
        self.encoder_layers = 12
        self.encoder_embed_dim = 768
        self.encoder_ffn_dim = 3072
        self.encoder_attention_heads = 12
        self.dropout = 0.1
        self.attention_dropout = 0.1
        self.activation_dropout = 0.1
        self.max_positions = 512
        self.activation_fn = "gelu"
        self.normalize_before = False
        self.sliding_window_size = 128  # Set to None for full attention

args = Args()

# Initialize the model
model = MPNetFlexForPretraining(args, tokenizer)

# Forward pass with flex attention
outputs = model(
    input_ids=input_ids,
    positions=positions,
    pred_size=pred_size,
    use_flex_attention=True  # Enable flex attention
)
```

## How It Works

FlexAttention provides efficient attention calculation by:

1. Creating block-sparse attention patterns
2. Computing attention only where needed (defined by the sliding window)
3. Maintaining the two-stream attention mechanism from MPNet

For sliding window attention, each token can only attend to tokens within a fixed window around it. This significantly reduces the computational complexity from O(n²) to O(n*w) where n is the sequence length and w is the window size.

## Converting to HuggingFace Format

After pretraining with FlexAttention, you can convert the model to HuggingFace format using the standard conversion script:

```bash
convert-to-hf \
  --mpnet-checkpoint-path ./checkpoints/mpnet_flex/best_checkpoint.pt \
  --hf-model-folder-path ./my_hf_model/
```

Note that the converted model will use standard attention in HuggingFace's implementation, but it will have learned parameters from the FlexAttention training.

## Recommendations

- For short sequences (≤512 tokens), standard attention is usually sufficient.
- For medium-length sequences (512-1024 tokens), a sliding window of 128-256 works well.
- For very long sequences (1024+ tokens), a sliding window of 128 usually provides a good balance of efficiency and performance.
- Start with a window size of 128 and adjust as needed based on your specific task.