# Model Architecture

This document describes the MPNet architecture implemented in `annotated-mpnet`.

## Overview

MPNet (Masked and Permuted Pre-training for Language Understanding) utilizes a **Masked and Permuted Pre-training** objective. The architecture is based on the Transformer model.

The pretraining objective involves predicting original tokens based on a permuted sequence where a subset of tokens has been masked. The permutation helps in learning richer contextual representations compared to standard Masked Language Modeling (MLM).

## Core Components

### MPNetForPretraining

The main model class defined in `annotated_mpnet/modeling/mpnet_for_pretraining.py`. It encapsulates the encoder and the language modeling head.

### SentenceEncoder

The core of the model, this is a stack of Transformer encoder layers. It's responsible for generating contextualized representations of the input tokens. Found in `annotated_mpnet/transformer_modules/sentence_encoder.py`.

### SentenceEncoderLayer

Each layer within the `SentenceEncoder`. It primarily consists of:

- **RelativeMultiHeadAttention**: A multi-head self-attention mechanism that incorporates relative positional information, crucial for MPNet. Defined in `annotated_mpnet/transformer_modules/rel_multihead_attention.py`.
- Position-wise Feed-Forward Networks (FFN).
- Layer normalization.

### Positional Embeddings

The model uses positional embeddings to provide sequence order information. This implementation supports:

- **LearnedPositionalEmbedding**: Positional embeddings are learned during training.
- **SinusoidalPositionalEmbedding**: Fixed positional embeddings based on sine and cosine functions.

The choice is configurable via `pretrain-mpnet` arguments. These are found in `annotated_mpnet/transformer_modules/`.

### MPNetLMHead

A language modeling head placed on top of the `SentenceEncoder`'s output. It projects the contextual embeddings to the vocabulary space to predict the masked tokens. Defined in `annotated_mpnet/modeling/mpnet_for_pretraining.py`.

## Two-Stream Self-Attention

A key innovation of MPNet. While not a separate module, this mechanism is implemented within the `MPNetForPretraining` forward pass. It allows the model to predict original tokens from a permuted version of the input by using two streams of information (content and query), enabling it to learn bidirectional context without the predicted tokens "seeing themselves" in the non-permuted context.

## Normalization Strategy

The `--normalize-before` flag (default: `False` in `SentenceEncoder`, `True` for `encoder_normalize_before` in `MPNetForPretraining`) controls whether layer normalization is applied before or after sublayer operations (attention and FFN), following common Transformer variations.
