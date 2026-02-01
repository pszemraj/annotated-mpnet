"""
Fairseq extension of LayerNorm which trys to use FusedLayerNorm if available
"""

from typing import Iterable, Union

import torch


def LayerNorm(
    normalized_shape: Union[int, Iterable[int], torch.Size],
    eps: float = 1e-5,
    elementwise_affine: bool = True,
    export: bool = False,
) -> torch.nn.Module:
    """Create a LayerNorm (fused if available).

    :param normalized_shape: Input shape to normalize.
    :param float eps: Epsilon for numerical stability, defaults to 1e-5.
    :param bool elementwise_affine: Whether to use affine parameters, defaults to True.
    :param bool export: Whether to disable fused layer norm for export, defaults to False.
    :return torch.nn.Module: LayerNorm module instance.
    """

    if not export and torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass

    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)
