"""
Module for defining the Encoder blocks in the transformer with FlexAttention support.
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
)
from annotated_mpnet.transformer_modules.flex_sentence_encoder_layer import FlexSentenceEncoderLayer


class FlexSentenceEncoder(nn.Module):
    """
    Implementation for a Bi-directional Transformer based Sentence Encoder with FlexAttention
    support for sliding window attention and other custom attention patterns.
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
        sliding_window_size: Optional[int] = None,
    ) -> None:
        """
        Initialize the FlexSentenceEncoder.

        Args:
            padding_idx: the index of the padding token
            vocab_size: the total number of tokens in vocab
            num_encoder_layers: how many SentenceEncoderLayers are in each SentenceEncoder
            embedding_dim: the dimension of the embeddings
            ffn_embedding_dim: the hidden size within the feed-forward network
            num_attention_heads: the number of attention heads in each layer
            dropout: the dropout prob for non-attention and non-activation layers
            attention_dropout: the dropout prob for the attention mechanism
            activation_dropout: the dropout prob inside the feed-forward network
            max_seq_len: the maximum number of tokens in a sequence
            num_segments: the number of segments within the input tokens
            use_position_embeddings: whether to use positional embeddings
            offset_positions_by_padding: whether to offset positions by padding_idx + 1
            encoder_normalize_before: whether to apply layer norm before attention
            activation_fn: the activation function used in the feed-forward network
            learned_pos_embedding: whether to use learned positional embeddings
            add_bias_kv: whether to add bias to K and V matrices
            add_zero_attn: whether to add zero attention
            embed_scale: scaling factor for token embeddings
            freeze_embeddings: whether to freeze embedding layers
            n_trans_layers_to_freeze: number of encoder layers to freeze
            relative_attention_num_buckets: number of buckets for relative attention
            normalize_before: whether to normalize before encoder layers
            export: whether to prepare for ONNX export
            sliding_window_size: the size of the sliding window for attention, None means full attention
        """

        super().__init__()

        # Store args
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.num_segments = num_segments
        self.use_position_embeddings = use_position_embeddings
        self.learned_pos_embedding = learned_pos_embedding
        self.sliding_window_size = sliding_window_size

        # Create the embedding layer
        self.embed_tokens = nn.Embedding(
            self.vocab_size, self.embedding_dim, self.padding_idx
        )

        # Store more args
        self.embed_scale = embed_scale

        # Get embeddings for token segment if num_segments > 0
        self.segment_embeddings = (
            nn.Embedding(self.num_segments, self.embedding_dim, padding_idx=None)
            if self.num_segments > 0
            else None
        )

        # Get positional embeddings
        self.embed_positions = (
            PositionalEmbedding(
                self.max_seq_len,
                self.embedding_dim,
                padding_idx=(self.padding_idx if offset_positions_by_padding else None),
                learned=self.learned_pos_embedding,
            )
            if self.use_position_embeddings
            else None
        )

        # Set up relative attention bias
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_bias = nn.Embedding(
            self.relative_attention_num_buckets, num_attention_heads, padding_idx=None
        )

        # Set up the encoder layers with FlexSentenceEncoderLayer
        self.layers = nn.ModuleList(
            [
                FlexSentenceEncoderLayer(
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
                    sliding_window_size=sliding_window_size,
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

        # Define a helper function to freeze layers
        def freeze_module_params(m: nn.Module):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        # Freeze embeddings if specified
        if freeze_embeddings:
            freeze_module_params(self.embed_tokens)
            freeze_module_params(self.segment_embeddings)
            freeze_module_params(self.embed_positions)
            freeze_module_params(self.emb_layer_norm)

        # Freeze encoder layers if specified
        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

    def forward(
        self,
        tokens: torch.Tensor,
        segment_labels: torch.Tensor = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
        use_flex_attention: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the FlexSentenceEncoder.

        Args:
            tokens: token indices
            segment_labels: segment labels for token type embeddings
            last_state_only: whether to return only the last state
            positions: optional position indices
            use_flex_attention: whether to use flex attention mechanism

        Returns:
            Tuple of (inner_states, sentence_rep) where inner_states contains the hidden
            states from all layers and sentence_rep is the representation of the first token
        """

        # Compute padding mask for attention
        padding_mask = tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None

        # Get the embeddings for the token sequence
        x = self.embed_tokens(tokens)

        # Scale the embeddings if specified
        if self.embed_scale is not None:
            x *= self.embed_scale

        # Add positional embeddings if specified
        if self.embed_positions is not None:
            x += self.embed_positions(tokens, positions=positions)

        # Add segment embeddings if specified
        if self.segment_embeddings is not None and segment_labels is not None:
            x += self.segment_embeddings(segment_labels)

        # Apply layer norm if specified
        if self.emb_layer_norm is not None and not self.normalize_before:
            x = self.emb_layer_norm(x)

        # Apply dropout
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Account for padding
        if padding_mask is not None:
            x *= 1 - padding_mask.unsqueeze(-1).type_as(x)

        # Transpose batch: B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # Compute relative attention bias
        positions_bias = self.compute_position_bias(
            x, self.relative_attention_num_buckets
        )

        # Track inner states if needed
        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        # Process through encoder layers
        for layer in self.layers:
            x, _ = layer(
                x, 
                self_attn_padding_mask=padding_mask, 
                positions_bias=positions_bias,
                use_flex_attention=use_flex_attention,
            )
            if not last_state_only:
                inner_states.append(x)

        # Apply final layer norm if specified
        if self.emb_layer_norm is not None and self.normalize_before:
            x = self.emb_layer_norm(x)

        # Transpose batch back: T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # Get sentence representation from first token
        sentence_rep = x[:, 0, :]

        # Add last state if needed
        if last_state_only:
            inner_states = [x]

        return inner_states, sentence_rep

    def compute_position_bias(self, x, num_buckets):
        """
        Helper function that computes the position bias based on the number of buckets.
        
        Args:
            x: input tensor
            num_buckets: number of buckets for relative position encoding
            
        Returns:
            Position bias tensor
        """
        # Get batch size, query length, and key length
        bsz, qlen, klen = x.size(1), x.size(0), x.size(0)
        
        # Create position indices
        context_position = torch.arange(qlen, dtype=torch.long)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long)[None, :]
        
        # Calculate relative positions
        relative_position = memory_position - context_position
        
        # Map relative positions to buckets
        rp_bucket = self.relative_position_bucket(
            relative_position, num_buckets=num_buckets
        )
        rp_bucket = rp_bucket.to(x.device)
        
        # Get relative position embeddings and reshape for attention
        values = self.relative_attention_bias(rp_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        values = values.expand((bsz, -1, qlen, klen)).contiguous()
        values = values.view(-1, qlen, klen)
        
        return values

    @staticmethod
    def relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        """
        Map relative positions to bucket indices.
        
        Args:
            relative_position: tensor of relative positions
            num_buckets: number of buckets to use
            max_distance: maximum distance for exponential buckets
            
        Returns:
            Bucket indices for the relative positions
        """
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