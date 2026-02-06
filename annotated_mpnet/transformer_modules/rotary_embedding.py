from __future__ import annotations

"""Rotary Position Embeddings (RoPE).

This implementation is intentionally minimal and repo-friendly:

- Works with *arbitrary* position ids (MPNet uses permuted positions).
- Applies RoPE to tensors shaped (B, H, L, D).
- Caches cos/sin tables (in fp32) and grows them on demand.

RoPE is applied to the first ``config.dim`` features of the head dimension and
leaves the remaining features untouched (useful if you want partial RoPE).

References:
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class RotaryConfig:
    """Configuration for RotaryEmbedding.

    Attributes:
        dim: Number of features in the head dimension to apply RoPE to. Must be even.
        base_theta: RoPE base ("theta"). Common values:
            - 10_000 (classic)
            - 160_000 (long context; used in some recent encoder configs)
        max_position_embeddings: Optional cap on the maximum position id + 1 allowed.
            If set, all position ids must be < max_position_embeddings.
    """

    dim: int
    base_theta: float = 10_000.0
    max_position_embeddings: Optional[int] = None


class RotaryEmbedding(nn.Module):
    """Applies RoPE to (B, H, L, D) tensors given arbitrary ``position_ids``.

    The cos/sin table is cached as fp32 on the module device and gathered by
    position ids (important for permuted position ids).
    """

    def __init__(self, config: RotaryConfig) -> None:
        super().__init__()
        if config.dim <= 0:
            raise ValueError(f"RotaryConfig.dim must be > 0, got {config.dim}.")
        if config.dim % 2 != 0:
            raise ValueError(f"RotaryConfig.dim must be even, got {config.dim}.")
        if config.base_theta <= 0:
            raise ValueError(f"RotaryConfig.base_theta must be > 0, got {config.base_theta}.")

        self.config = config

        inv_freq = 1.0 / (
            config.base_theta
            ** (torch.arange(0, config.dim, 2, dtype=torch.float32) / float(config.dim))
        )
        # (dim/2,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cached tables (fp32): (max_pos, dim/2)
        self.register_buffer("_cos_cached", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer("_sin_cached", torch.empty(0, dtype=torch.float32), persistent=False)
        self._cached_max_pos: int = 0
        self._cache_device: Optional[torch.device] = None

    @torch.no_grad()
    def _maybe_build_cache(self, *, max_pos: int, device: torch.device) -> None:
        """Ensure cache supports position ids in [0, max_pos)."""
        if max_pos <= 0:
            return

        if (
            self.config.max_position_embeddings is not None
            and max_pos > self.config.max_position_embeddings
        ):
            raise ValueError(
                f"RoPE cache request max_pos={max_pos} exceeds max_position_embeddings={self.config.max_position_embeddings}."
            )

        # If cache is already large enough and on the right device, keep it.
        if self._cache_device == device and self._cached_max_pos >= max_pos:
            return

        # Build [0..max_pos-1] table.
        t = torch.arange(max_pos, device=device, dtype=torch.float32)  # (max_pos,)
        freqs = torch.outer(t, self.inv_freq.to(device=device))  # (max_pos, dim/2)

        self._cos_cached = freqs.cos()
        self._sin_cached = freqs.sin()
        self._cached_max_pos = int(max_pos)
        self._cache_device = device

    def _gather_cos_sin(
        self,
        position_ids: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather cos/sin at (B, L) position ids.

        Returns:
            cos, sin of shape (B, L, dim/2) in requested dtype.
        """
        if position_ids.dim() != 2:
            raise ValueError(
                f"position_ids must have shape (B, L), got {tuple(position_ids.shape)}."
            )

        # Move/cast positions to the same device as the tensor being rotated.
        if position_ids.device != device:
            position_ids = position_ids.to(device)
        if position_ids.dtype not in (torch.int32, torch.int64):
            position_ids = position_ids.to(torch.int64)

        if position_ids.numel() == 0:
            # Degenerate but safe.
            b, seq = position_ids.shape
            empty = torch.empty((b, seq, self.config.dim // 2), device=device, dtype=dtype)
            return empty, empty

        max_pos = int(position_ids.max().item()) + 1
        self._maybe_build_cache(max_pos=max_pos, device=device)

        # (B*L,) -> gather -> (B, L, dim/2)
        flat = position_ids.reshape(-1)
        cos = self._cos_cached.index_select(0, flat).view(*position_ids.shape, -1).to(dtype=dtype)
        sin = self._sin_cached.index_select(0, flat).view(*position_ids.shape, -1).to(dtype=dtype)
        return cos, sin

    def rotate(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to x.

        Args:
            x: Tensor shaped (B, H, L, D).
            position_ids: Tensor shaped (B, L), can be arbitrary integers >= 0.

        Returns:
            Tensor shaped (B, H, L, D) with RoPE applied on the first ``config.dim`` features.
        """
        if x.dim() != 4:
            raise ValueError(f"x must have shape (B, H, L, D), got {tuple(x.shape)}.")

        b, h, seq_len, d = x.shape
        if position_ids.shape != (b, seq_len):
            raise ValueError(
                f"position_ids must have shape (B, L) == ({b}, {seq_len}), got {tuple(position_ids.shape)}."
            )

        rotary_dim = self.config.dim
        if rotary_dim > d:
            raise ValueError(f"RotaryConfig.dim={rotary_dim} cannot exceed head_dim={d}.")

        # Gather cos/sin and broadcast over heads.
        cos, sin = self._gather_cos_sin(position_ids, device=x.device, dtype=x.dtype)
        cos = cos.unsqueeze(1)  # (B, 1, L, dim/2)
        sin = sin.unsqueeze(1)

        x_rope = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]

        # (B, H, L, dim) -> (B, H, L, dim/2, 2)
        x_rope = x_rope.view(b, h, seq_len, rotary_dim // 2, 2)
        x1 = x_rope[..., 0]
        x2 = x_rope[..., 1]

        # Apply complex rotation:
        # (x1 + i*x2) * (cos + i*sin) => (x1*cos - x2*sin) + i*(x1*sin + x2*cos)
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        out = torch.stack((out1, out2), dim=-1).reshape(b, h, seq_len, rotary_dim)

        if x_pass.numel() == 0:
            return out
        return torch.cat((out, x_pass), dim=-1)

    def rotate_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_position_ids: torch.Tensor,
        k_position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to q and k (optionally with different position ids)."""
        if k_position_ids is None:
            k_position_ids = q_position_ids
        return self.rotate(q, q_position_ids), self.rotate(k, k_position_ids)
