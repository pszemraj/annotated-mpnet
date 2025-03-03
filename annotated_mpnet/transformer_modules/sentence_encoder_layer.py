"""
Module for defining the encoder sublayer with better torch.compile compatibility
"""

import logging
from typing import Tuple, Optional

from rich.logging import RichHandler

LOG_FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=LOG_FORMAT, datefmt="[%X] ", handlers=[RichHandler()]
)
LOGGER = logging.getLogger(__name__)


import torch
import torch.nn.functional as F
from torch import nn

from annotated_mpnet.transformer_modules import LayerNorm, RelativeMultiHeadAttention
from annotated_mpnet.utils import utils


class SentenceEncoderLayer(nn.Module):
    """
    Implements the the layers within the full SentenceEncoder
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        normalize_before: bool = True,
        export: bool = False,
        use_flex_attention: bool = False,
        sliding_window_size: Optional[int] = None,
    ) -> None:
        """
        Init function for the layer with torch.compile optimizations
        """
        super().__init__()

        # Store args
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.use_flex_attention = use_flex_attention
        self.sliding_window_size = sliding_window_size
        self.num_heads = num_attention_heads  # Store for consistent access

        # Get the submodules we need
        self.activation_fn = utils.get_activation_fn(activation_fn)

        # Initialize the appropriate attention mechanism
        if not use_flex_attention:
            # Use standard RelativeMultiHeadAttention
            self.self_attn = RelativeMultiHeadAttention(
                self.embedding_dim,
                num_attention_heads,
                dropout=attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=True,
            )
        else:
            # For FlexAttention, we'll lazily initialize it in forward
            # This avoids circular imports and makes tracing easier
            self._flex_attention = None
            self.num_attention_heads = num_attention_heads
            self.attention_dropout = attention_dropout

        # Get the LayerNorm for the self_attention output
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        # Get the FC linear layers for the hidden connections
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # Get the final LayerNorm
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

        self.normalize_before = normalize_before

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        positions_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass optimized for torch.compile
        """
        # Store residual for skip connection
        residual = x

        # Apply layer norm if needed
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)

        # Choose appropriate attention implementation
        if self.use_flex_attention:
            # Defer the import to avoid circular imports
            from annotated_mpnet.transformer_modules.flex_attention import FlexTwoStreamAttention
            
            # Create FlexAttention module if it doesn't exist yet
            if self._flex_attention is None:
                self._flex_attention = FlexTwoStreamAttention(
                    embed_dim=self.embedding_dim,
                    num_heads=self.num_attention_heads,
                    dropout=self.attention_dropout,
                    use_sliding_window=self.sliding_window_size is not None,
                    sliding_window_size=self.sliding_window_size or 0,
                ).to(x.device)
            
            # Use FlexAttention
            # For standard self-attention, content and query are the same
            (x, _), attn = self._flex_attention(
                (x, x),  # content and query are the same
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                positions_bias=positions_bias,
                need_weights=False,
            )
        else:
            # Use standard attention
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
                positions_bias=positions_bias,
            )

        # Process dropout and skip connection
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        # Process feed-forward network
        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + residual
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)

        return x, attn

    def maybe_layer_norm(
        self, layer_norm: nn.Module, x: torch.Tensor, before: bool = False, after: bool = False
    ) -> torch.Tensor:
        """
        Helper function for conditional layer normalization
        """
        assert before ^ after, "You must set only one of 'before' or 'after'"

        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x