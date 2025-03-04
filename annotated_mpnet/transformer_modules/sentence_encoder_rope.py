"""
Module for defining the Encoder blocks in the transformer with RoPE support
"""

import logging
from typing import Optional, Tuple

from rich.logging import RichHandler

LOG_FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=LOG_FORMAT, datefmt="[%X] ", handlers=[RichHandler()]
)
LOGGER = logging.getLogger(__name__)


import math

import torch
import torch.nn.functional as F
from torch import nn

from annotated_mpnet.transformer_modules import (
    LayerNorm,
    PositionalEmbedding,
    SentenceEncoderLayer,
    RoPERelativeMultiHeadAttention,
)


class SentenceEncoderRoPE(nn.Module):
    """
    Implementation for a Bi-directional Transformer based Sentence Encoder used
    in BERT/XLM style pre-trained models, with RoPE support.
    This first computes the token embedding using the token embedding matrix,
    and can optionally use traditional position embeddings or RoPE.
    After applying the specified number of TransformerEncoderLayers, it outputs 
    all the internal states of the encoder as well as the final representation 
    associated with the first token (usually CLS token).
    """

    def __init__(
        self,
        padding_idx: int,
        vocab_size: int,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        max_seq_len: int = 256,
        num_segments: int = 2,
        use_position_embeddings: bool = True,
        offset_positions_by_padding: bool = True,
        encoder_normalize_before: bool = False,
        activation_fn: str = "relu",
        learned_pos_embedding: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        relative_attention_num_buckets: int = 32,
        normalize_before: bool = False,
        export: bool = False,
        use_rope: bool = False,
        rope_base: int = 10000,
        rope_scaling: float = 1.0,
    ) -> None:
        """
        Initialize the SentenceEncoder with RoPE option

        Args:
            Additional args beyond standard SentenceEncoder:
            use_rope: Whether to use Rotary Position Embeddings instead of traditional embeddings
            rope_base: Base for RoPE frequency bands
            rope_scaling: Scaling factor for RoPE (useful for longer sequences)
        """

        super().__init__()

        # Store all args
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.num_segments = num_segments
        self.use_position_embeddings = use_position_embeddings and not use_rope
        self.learned_pos_embedding = learned_pos_embedding
        self.use_rope = use_rope

        # Create the embedding layer that will convert token IDs into embeds
        self.embed_tokens = nn.Embedding(
            self.vocab_size, self.embedding_dim, self.padding_idx
        )

        # Store more args
        self.embed_scale = embed_scale

        # Get embeddings for token segment. Only created if num_segments > 0
        self.segment_embeddings = (
            nn.Embedding(self.num_segments, self.embedding_dim, padding_idx=None)
            if self.num_segments > 0
            else None
        )

        # Get positional embeddings, only if not using RoPE
        if not use_rope and self.use_position_embeddings:
            self.embed_positions = PositionalEmbedding(
                self.max_seq_len,
                self.embedding_dim,
                padding_idx=(self.padding_idx if offset_positions_by_padding else None),
                learned=self.learned_pos_embedding,
            )
        else:
            self.embed_positions = None

        # Set up relative attention bias for the attention mechanism
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_bias = nn.Embedding(
            self.relative_attention_num_buckets, num_attention_heads, padding_idx=None
        )

        # Set up the encoder layers in the typical way using a module list
        # Use RoPE attention if configured, otherwise use standard attention
        self.layers = nn.ModuleList(
            [
                SentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    add_bias_kv=add_bias_kv,
                    add_zero_attn=add_zero_attn,
                    normalize_before=normalize_before,
                    export=export,
                    use_rope=use_rope,
                    max_position_embeddings=max_seq_len,
                    rope_base=rope_base,
                    rope_scaling=rope_scaling,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # Set up the layer norm
        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        self.normalize_before = normalize_before

        # Define a helper function to freeze embedding layers if specified in the args
        def freeze_module_params(m: nn.Module):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        # Now use the helper function to freeze params if specified
        if freeze_embeddings:
            freeze_module_params(self.embed_tokens)
            freeze_module_params(self.segment_embeddings)
            freeze_module_params(self.embed_positions)
            freeze_module_params(self.emb_layer_norm)

        # We can also freeze encoder layers with the n_trans_layers_to_freeze which we process here
        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

    def forward(
        self,
        tokens: torch.Tensor,
        segment_labels: torch.Tensor = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward pass of the Encoder
        """

        # Compute padding mask. This is needed for multi-head attention
        padding_mask = tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None

        # Get the embeddings for the token sequence
        x = self.embed_tokens(tokens)

        # Scale the embeddings if the appropriate arg is specified
        if self.embed_scale is not None:
            x *= self.embed_scale

        # Add in positional embeddings if they are specified and we're not using RoPE
        if self.embed_positions is not None:
            x += self.embed_positions(tokens, positions=positions)

        # If there is a segment label, pass those segments into the segment_embedding layer
        if self.segment_embeddings is not None and segment_labels is not None:
            x += self.segment_embeddings(segment_labels)

        # Process the layer norm
        if self.emb_layer_norm is not None and not self.normalize_before:
            x = self.emb_layer_norm(x)

        # Dropout after the layer norm
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Account for padding while computing the representation
        if padding_mask is not None:
            x *= 1 - padding_mask.unsqueeze(-1).type_as(x)

        # Transpose the batch for easier attention calculation later on. This is an artifact of the
        # fairseq codebase, but since it's done like this everywhere, we have to keep it
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # Compute the relative attention bias
        positions_bias = self.compute_position_bias(
            x, self.relative_attention_num_buckets
        )

        # If the user wants ALL hidden states, we keep track of it here
        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        # Now process through all the encoder layers (and add each intermediate state if
        # last_state_only is False)
        for layer in self.layers:
            x, _ = layer(
                x, self_attn_padding_mask=padding_mask, positions_bias=positions_bias
            )
            if not last_state_only:
                inner_states.append(x)

        # Compute the layer norm if the bools evaluate properly
        if self.emb_layer_norm is not None and self.normalize_before:
            x = self.emb_layer_norm(x)

        # Transpose the batch back to the standard format
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # Get the sentence representation by extracting the CLS token embedding (index 0)
        sentence_rep = x[:, 0, :]

        # If the user only wants the last state only, here's where we add it
        if last_state_only:
            inner_states = [x]

        return inner_states, sentence_rep

    # Helper function below
    def compute_position_bias(self, x, num_buckets):
        """
        Helper function that computes the position bias based on the number of buckets provided
        """

        # Get the batch size, q and k len
        bsz, qlen, klen = x.size(1), x.size(0), x.size(0)
        context_position = torch.arange(qlen, dtype=torch.long)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long)[None, :]

        relative_position = memory_position - context_position

        rp_bucket = self.relative_position_bucket(
            relative_position, num_buckets=num_buckets
        )
        rp_bucket = rp_bucket.to(x.device)
        values = self.relative_attention_bias(rp_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        values = values.expand((bsz, -1, qlen, klen)).contiguous()
        values = values.view(-1, qlen, klen)
        return values

    @staticmethod
    def relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).to(torch.long) * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)

        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )
        ret += torch.where(is_small, n, val_if_large)
        return ret