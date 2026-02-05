"""
Utility helpers for tensor ops and optional handling.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Iterable, Optional, TypeVar

import torch

T = TypeVar("T")


def exists(value: Any) -> bool:
    """Return True when a value is not None.

    :param Any value: Value to check.
    :return bool: True when ``value`` is not None.
    """
    return value is not None


def maybe(fn: Callable[..., T]) -> Callable[..., Optional[T]]:
    """Decorate a function to short-circuit when its first argument is None.

    :param Callable fn: Callable to wrap.
    :return Callable: Wrapped callable returning None when the first arg is None.
    """

    @wraps(fn)
    def inner(value: Optional[T], *args: Any, **kwargs: Any) -> Optional[T]:
        if not exists(value):
            return None
        return fn(value, *args, **kwargs)

    return inner


def pad_left_ndim_to(tensor: torch.Tensor, ndims: int) -> torch.Tensor:
    """Pad tensor with leading singleton dimensions until it has ``ndims`` dims.

    :param torch.Tensor tensor: Input tensor to reshape.
    :param int ndims: Target number of dimensions.
    :return torch.Tensor: Reshaped tensor with leading singleton dims.
    """
    if tensor.ndim >= ndims:
        return tensor
    shape = (1,) * (ndims - tensor.ndim) + tuple(tensor.shape)
    return tensor.reshape(*shape)


def normalize_position_bias(
    bias: torch.Tensor,
    bsz: int,
    num_heads: int,
    tgt_len: int,
    src_len: int,
    device: torch.device,
    dtype: torch.dtype,
    expand_batch: bool = False,
) -> torch.Tensor:
    """Normalize position bias to 4D (batch, heads, tgt, src) format.

    :param torch.Tensor bias: Position bias tensor to normalize.
    :param int bsz: Batch size.
    :param int num_heads: Number of attention heads.
    :param int tgt_len: Target sequence length.
    :param int src_len: Source sequence length.
    :param torch.device device: Device to match.
    :param torch.dtype dtype: Data type to match.
    :param bool expand_batch: Whether to expand batch dim when bias has batch size 1.
    :return torch.Tensor: Normalized bias tensor with shape (bsz|1, heads, tgt, src).
    """
    if bias.device != device or bias.dtype != dtype:
        bias = bias.to(device=device, dtype=dtype)

    if bias.dim() == 3:
        if bias.size(0) == bsz * num_heads:
            bias = bias.view(bsz, num_heads, tgt_len, src_len)
        elif bias.size(0) != num_heads:
            raise ValueError(
                "positions_bias has unexpected shape; expected heads or bsz*heads in dim 0."
            )
    elif bias.dim() == 4:
        if bias.size(1) != num_heads or bias.size(0) not in (1, bsz):
            raise ValueError(
                "positions_bias has unexpected shape; expected (1|bsz, heads, tgt, src)."
            )
    else:
        raise ValueError("positions_bias must be 3D or 4D for attention biasing.")

    bias = pad_left_ndim_to(bias, 4)

    if expand_batch and bias.size(0) == 1 and bsz > 1:
        bias = bias.expand(bsz, -1, -1, -1).contiguous()

    return bias


def compact(values: Iterable[Optional[T]]) -> list[T]:
    """Drop None entries from an iterable.

    :param Iterable[Optional[T]] values: Iterable of optional values.
    :return list[T]: List with None entries removed.
    """
    return [value for value in values if value is not None]


def reduce_masks(
    masks: Iterable[Optional[torch.Tensor]],
    op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Optional[torch.Tensor]:
    """Reduce masks with a binary logical op.

    :param Iterable[Optional[torch.Tensor]] masks: Mask tensors to combine.
    :param Callable op: Binary op to combine masks (e.g., torch.logical_or).
    :return Optional[torch.Tensor]: Combined mask or None when ``masks`` is empty.
    """
    mask_list = compact(masks)
    if not mask_list:
        return None
    mask, *rest = mask_list
    for other in rest:
        mask = op(mask, other)
    return mask


def or_masks(masks: Iterable[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    """Combine masks with logical OR.

    :param Iterable[Optional[torch.Tensor]] masks: Mask tensors to combine.
    :return Optional[torch.Tensor]: Combined mask or None when empty.
    """
    return reduce_masks(masks, torch.logical_or)


def and_masks(masks: Iterable[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    """Combine masks with logical AND.

    :param Iterable[Optional[torch.Tensor]] masks: Mask tensors to combine.
    :return Optional[torch.Tensor]: Combined mask or None when empty.
    """
    return reduce_masks(masks, torch.logical_and)
