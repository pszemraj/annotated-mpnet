"""
Wrapping function for positional embeddings which allows users to selected learned or sinusoidal
embeddings
"""

import logging

from rich.logging import RichHandler

LOG_FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=LOG_FORMAT, datefmt="[%X] ", handlers=[RichHandler()])
LOGGER = logging.getLogger(__name__)


from torch import nn

from annotated_mpnet.transformer_modules import (
    LearnedPositionalEmbedding,
    SinusoidalPositionalEmbedding,
)


def PositionalEmbedding(
    num_embeddings: int, embedding_dim: int, padding_idx: int, learned: bool = False
) -> nn.Module:
    """Create positional embedding module.

    :param int num_embeddings: Number of embeddings.
    :param int embedding_dim: Embedding dimensionality.
    :param int padding_idx: Padding index.
    :param bool learned: Whether to use learned embeddings, defaults to False.
    :return nn.Module: Positional embedding module.
    """

    # If we specified "learned" to be True, we want to create a learned positional embedding module
    if learned:
        num_embeddings = num_embeddings + 2  # Add 2 for CLS and SEP

        # Instantiate the learned positional embeddings
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)

        # Make sure the weights are properly initialized here
        nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)

        # If we specified a padding index, we need to make sure this weight is zeroed out
        if padding_idx is not None:
            nn.init.constant_(m.weight[padding_idx], 0)
    # Branch to create sinusoidal embeddings if "learned" is False
    else:
        m = SinusoidalPositionalEmbedding(
            embedding_dim,
            padding_idx,
            init_size=num_embeddings + 2,  # Add 2 for CLS and SEP
        )

    return m
