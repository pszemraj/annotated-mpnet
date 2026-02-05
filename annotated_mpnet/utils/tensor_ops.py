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
