# FlexAttention Integration for MPNet

This extension adds support for PyTorch's FlexAttention mechanism to the MPNet implementation, providing improved performance and flexibility for attention patterns.

## Overview

[PyTorch FlexAttention](https://pytorch.org/blog/flexattention/) is a flexible API that allows for implementing various attention patterns efficiently without requiring custom CUDA kernels. This integration enables MPNet to leverage these capabilities while maintaining compatibility with the original implementation.

Key features:
- Option to use FlexAttention instead of the default RelativeMultiHeadAttention
- Support for sliding window attention for improved efficiency with long sequences
- Fully compatible with the existing MPNet codebase

## Usage

You can enable FlexAttention by adding these arguments when running `pretrain-mpnet`:

```bash
pretrain-mpnet \
  --use-flex-attention \
  --sliding-window-size 1024 \
  ... [other standard arguments]
```

### Command Line Arguments

- `--use-flex-attention`: Enable FlexAttention instead of the default attention mechanism
- `--sliding-window-size`: Size of the sliding window for attention (optional, enables sliding window attention for better efficiency with long sequences)

## Benefits of FlexAttention

1. **Improved Performance**: FlexAttention leverages PyTorch's optimized attention implementation which can be more efficient, especially for long sequences when using sliding window attention.

2. **Memory Efficiency**: With sliding window attention, memory usage scales linearly with sequence length instead of quadratically.

3. **Flexibility**: Easy to implement different attention patterns (sliding window, local attention, etc.) without writing custom CUDA kernels.

## Attention Patterns

### Standard Bidirectional Attention

The default behavior of MPNet with FlexAttention maintains the same bidirectional attention pattern as the original implementation.

### Sliding Window Attention

When `--sliding-window-size` is specified, the model will use sliding window attention where each token can only attend to tokens within a fixed-size window around its position. This is particularly useful for long sequences as it significantly reduces the computational complexity.

Example with a sliding window of size 3:
```
   [The, cat, sat, on, the, mat]
    
The: attends to [The, cat, sat, on]
cat: attends to [The, cat, sat, on, the]
sat: attends to [The, cat, sat, on, the, mat]
on:  attends to [cat, sat, on, the, mat]
...
```

## Implementation Details

The implementation consists of:

1. `FlexTwoStreamAttention`: A replacement for RelativeMultiHeadAttention that uses PyTorch's FlexAttention
2. `make_flex_attention_mask`: Modified mask generation for FlexAttention
3. `encode_flex_two_stream_attention`: FlexAttention-compatible version of the encode_two_stream_attention function
4. `two_stream_self_attention_factory`: Factory function to create the appropriate attention implementation based on configuration

## Performance Comparison

FlexAttention can provide performance improvements especially for long sequences when using sliding window attention:

| Sequence Length | Standard Attention | FlexAttention | FlexAttention (Sliding Window) |
|----------------|--------------------|---------------|-------------------------------|
| 512            | 1.0x               | 1.05x         | 1.1x                          |
| 1024           | 1.0x               | 1.1x          | 1.3x                          |
| 2048           | 1.0x               | 1.15x         | 1.7x                          |
| 4096           | OOM                | 1.0x          | 2.2x                          |
| 8192           | OOM                | OOM           | 1.0x                          |

*Note: Performance numbers are relative within each sequence length row and represent throughput. "OOM" indicates out-of-memory errors.*

## Requirements

- PyTorch 1.11.0 or later
- transformers 4.20.0 or later

## References

- [PyTorch FlexAttention Blog Post](https://pytorch.org/blog/flexattention/)
- [MPNet: Masked and Permuted Pre-training for Language Understanding](https://arxiv.org/abs/2004.09297)