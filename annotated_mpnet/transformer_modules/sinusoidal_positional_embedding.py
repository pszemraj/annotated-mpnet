"""
Module containing the SinusoidalPositionalEmbedding option, which creates a sinusoidal relationship
between a token and its position
"""

import logging
from typing import Any, Dict, Optional

LOGGER = logging.getLogger(__name__)


import math

import torch
import torch.onnx.operators
from torch import nn

from annotated_mpnet.utils import utils


class SinusoidalPositionalEmbedding(nn.Module):
    """
    A module for creating positional embeddings that follow a sinusoidal relationship
    """

    def __init__(self, embedding_dim: int, padding_idx: int, init_size: int = 1024) -> None:
        """Initialize sinusoidal positional embeddings.

        :param int embedding_dim: Embedding dimensionality.
        :param int padding_idx: Padding index.
        :param int init_size: Initial size of the embedding table, defaults to 1024.
        """
        super().__init__()

        # Store args
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # Get the weights from a helper function that processes the sinusoidal math
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )

        # Set the ONNX trace variable again, but I don't think we'll be using it
        self.onnx_trace = False

        # This is a builtin nn.Module function for registering buffer values within a module
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    def prepare_for_onnx_export(self) -> None:
        """Prepare the module for ONNX export."""
        self.onnx_trace = True

    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ) -> torch.Tensor:
        """Instantiate sinusoidal positional embedding weights.

        :param int num_embeddings: Number of embeddings.
        :param int embedding_dim: Embedding dimensionality.
        :param int padding_idx: Padding index, defaults to None.
        :return torch.Tensor: Embedding weight matrix.
        """

        # First get half the embedding dimension size
        half_dim = embedding_dim // 2

        # Not quite sure of the math happening below, but generally, we are constructing initial
        # weights using an algorithm that heavily involves trigonometric relationships (as you can
        # see in the last step with sin() and cos() making an appearance)
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)

        # Next calculate padding. If embedding size is not divisible by 2, we need to pad out
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)

        # If there IS a padding index, reset the weights to 0
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(
        self,
        input: torch.Tensor,
        incremental_state: Optional[Dict[str, Any]] = None,
        timestep: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute sinusoidal positional embeddings.

        :param torch.Tensor input: Input tensor of shape (bsz, seq_len).
        :param dict incremental_state: Incremental decoding state, defaults to None.
        :param torch.Tensor timestep: Timestep tensor for incremental decoding, defaults to None.
        :param dict kwargs: Additional unused keyword arguments.
        :return torch.Tensor: Positional embeddings.
        """

        # Break out dimensions
        bsz, seq_len = torch.onnx.operators.shape_as_tensor(input)

        # Get the max position of the given sequence
        max_pos = (self.padding_idx + 1) + seq_len

        # Now we add the option to recompute embeddings if the initial embeddings weren't large
        # enough to cover all positons
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )

        self.weights = self.weights.to(self._float_tensor)

        # Process incremental state below
        # Again, not really sure what this does
        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            if self.onnx_trace:
                return (
                    self.weights.index_select(index=self.padding_idx + pos, dim=0)
                    .unsqueeze(1)
                    .repeat(bsz, 1, 1)
                )
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        # Use the typical `make_positions` util to get incremental positions. This will eventually
        # feed directly into the sinusoidal weights we defined before
        positions = utils.make_positions(input, self.padding_idx, onnx_trace=self.onnx_trace)

        # If onnx_trace is set (which it shouldn't be), process additional below
        if self.onnx_trace:
            flat_embeddings = self.weights.detach().index_select(0, positions.view(-1))
            embedding_shape = torch.cat((bsz.view(1), seq_len.view(1), torch.LongTensor([-1])))
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(
                flat_embeddings, embedding_shape
            )
            return embeddings

        # Return the weights selected by the positions generated above
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    # Helper function below
    def max_positions(self) -> int:
        """Return the maximum number of supported positions.

        :return int: Maximum supported positions.
        """
        return int(1e5)  # an arbitrary large number
