"""Project-wide constants used across modules."""

from typing import Optional


def position_offset(padding_idx: Optional[int]) -> int:
    """Return the position offset implied by a padding index.

    :param Optional[int] padding_idx: Padding index for positional embeddings.
    :return int: Offset applied to 0-based positions to align with make_positions.
    """
    if padding_idx is None:
        return 0
    return padding_idx + 1
