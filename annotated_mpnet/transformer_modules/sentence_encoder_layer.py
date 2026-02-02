"""
Module for defining the encoder sublayer. This will eventually wrap into the full SentenceEncoder
class
"""

import logging
from typing import Optional, Tuple

from rich.logging import RichHandler

LOG_FORMAT = "%(message)s"
# NOTE: basicConfig is a no-op if logging is already configured by the host app.
logging.basicConfig(level="INFO", format=LOG_FORMAT, datefmt="[%X] ", handlers=[RichHandler()])
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
    ) -> None:
        """Initialize a sentence encoder layer.

        :param float embedding_dim: Embedding dimension, defaults to 768.
        :param float ffn_embedding_dim: FFN hidden size, defaults to 3072.
        :param float num_attention_heads: Number of attention heads, defaults to 8.
        :param float dropout: Dropout probability, defaults to 0.1.
        :param float attention_dropout: Attention dropout probability, defaults to 0.1.
        :param float activation_dropout: Activation dropout probability, defaults to 0.1.
        :param str activation_fn: Activation function name, defaults to "relu".
        :param bool add_bias_kv: Whether to add bias to K/V, defaults to False.
        :param bool add_zero_attn: Whether to add zero attention, defaults to False.
        :param bool normalize_before: Normalize before attention, defaults to True.
        :param bool export: Whether to enable ONNX tracing, defaults to False.
        """
        super().__init__()

        # Store args
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Get the submodules we need
        self.activation_fn = utils.get_activation_fn(activation_fn)

        # Initialize the self attention module
        self.self_attn = RelativeMultiHeadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True,
        )

        # Get the LayerNorm for the self_attention output
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        # Get the FC linear layers for the hidden connections in each layer after the self-attention
        # is calculated
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # Get the final LayerNorm
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

        self.normalize_before = normalize_before

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        positions_bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the encoder layer forward pass.

        :param torch.Tensor x: Input tensor.
        :param torch.Tensor self_attn_mask: Self-attention mask, defaults to None.
        :param torch.Tensor self_attn_padding_mask: Padding mask, defaults to None.
        :param torch.Tensor positions_bias: Relative position bias, defaults to None.
        :return Tuple[torch.Tensor, Optional[torch.Tensor]]: Output tensor and attention weights.
        """

        # Keep the residual for the skip connection after self-attention calculation
        residual = x

        # This is a bit of an overloaded function that will check if normalization should be
        # processed before or after self-attention. It will cross reference the "before" or "after"
        # kwarg against self.normalize_before arg and then either do nothing or normalize
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)

        # Forward pass of self-attention
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
            positions_bias=positions_bias,
        )

        # The below operations may look scary, but we will do our best to summarize their use

        # Process the dropout after self-attention is calcualted and then make the skip connection
        # by adding residual to x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        # Try the maybe_layer_norm function again to potentially normalize after self-attention
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        # Now we must process the fully connected layer after self-attention. Similarly, there is
        # also a LayerNorm that must be calculated before or after (and is determined by the
        # normalize_before arg)

        # Save the residual for the skip connection after FC layer
        residual = x

        # Process the layer norm
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)

        # Process the first layer of the feed-forward network which expands the embedding from
        # embedding_dim to ffn_embedding_dim (Linear + activation)
        x = self.activation_fn(self.fc1(x))

        # Process the dropout once again
        x = F.dropout(x, p=self.activation_dropout, training=self.training)

        # Calculate the second portion of the feed-forward net, converting the hidden size back to
        # our embedding size of embedding_dim. This time we DO NOT add the activation function so
        # as to not kill the neurons
        x = self.fc2(x)

        # Process the droput again
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Calculate the skip connection with the residual and the output of the feed-forward net
        x = x + residual

        # Finally, process the LayerNorm once again
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)

        return x, attn

    def maybe_layer_norm(
        self,
        layer_norm: nn.Module,
        x: torch.Tensor,
        before: bool = False,
        after: bool = False,
    ) -> torch.Tensor:
        """Conditionally apply layer normalization.

        :param nn.Module layer_norm: Layer norm module.
        :param torch.Tensor x: Input tensor.
        :param bool before: Whether called before attention, defaults to False.
        :param bool after: Whether called after attention, defaults to False.
        :return torch.Tensor: Normalized tensor (or original if skipped).
        """
        # First make sure before and after both aren't true with a quick XOR
        assert before ^ after, "You must set only one of 'before' or 'after'"

        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x
