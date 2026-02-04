"""
Module containing the LearnedPositionalEmbedding option, which learns position values instead of
something like a sinusoidal distribution
"""

import logging
from typing import Any, Dict, Optional

LOGGER = logging.getLogger(__name__)


import torch
from torch import nn

from annotated_mpnet.utils import utils


class LearnedPositionalEmbedding(nn.Embedding):
    """
    A subclass of the Embedding module that will operate as a layer for learning positional embeds
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int) -> None:
        """Initialize learned positional embeddings.

        :param int num_embeddings: Number of embeddings.
        :param int embedding_dim: Embedding dimensionality.
        :param int padding_idx: Padding index.
        """
        # Initialize the superclass embedding layer
        super().__init__(num_embeddings, embedding_dim, padding_idx)

        # We set this ONNX variable just in case it breaks something down the line, but I think it's
        # useless for us
        self.onnx_trace = False

    def forward(
        self,
        input: torch.Tensor,
        incremental_state: Optional[Dict[str, Any]] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute learned positional embeddings.

        :param torch.Tensor input: Input batch of shape (bsz, seq_len).
        :param dict incremental_state: Incremental decoding state, defaults to None.
        :param torch.Tensor positions: Precomputed positions, defaults to None.
        :return torch.Tensor: Positional embeddings.
        """

        # Assert that only one of `positions` or `padding_idx` is set
        assert (positions is None) or (self.padding_idx is None), (
            "If `positions` is precomputed, do NOT pass in a padding_idx"
        )

        # Let's create the positions if they are not precomputed
        if positions is None:
            # We branch to this "incremental_state" logic only if we're doing ONNX exporting
            if incremental_state is not None:
                # positions is the same for every token when decoding a single step
                # Without the int() cast, it doesn't work in some cases when exporting to ONNX
                positions = input.data.new(1, 1).fill_(int(self.padding_idx + input.size(1)))
            else:
                # Create positions using the `make_positions` function. This basically just creates
                # incremental positions starting at padding_idx+1. Very simple function that you
                # can check in the utils package
                positions = utils.make_positions(
                    input.data,
                    self.padding_idx,
                    onnx_trace=self.onnx_trace,
                )

        # Do the actual embedding pass here now
        return super().forward(positions)

    # Below are some convenience functions and aliases for this class. Should not be of too much
    # importance for our usage
    def max_positions(self) -> int:
        """Return the maximum number of positional embeddings.

        :return int: Maximum supported positions.
        """
        if self.padding_idx is not None:
            return self.num_embeddings - self.padding_idx - 1
        else:
            return self.num_embeddings

    def _forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Alias for calling the embedding layer with precomputed positions.

        :param torch.Tensor positions: Precomputed position indices.
        :return torch.Tensor: Positional embeddings.
        """
        return super().forward(positions)
