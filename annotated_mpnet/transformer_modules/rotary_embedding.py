from __future__ import annotations

"""Rotary Position Embeddings (RoPE).

This implementation is intentionally minimal and repo-friendly:

- Works with *arbitrary* position ids (MPNet uses permuted positions).
- Applies RoPE to tensors shaped (B, H, L, D).
- Caches cos/sin tables (in fp32) and grows them on demand.

RoPE is applied to the first ``config.dim`` features of the head dimension and
leaves the remaining features untouched (useful if you want partial RoPE).

When ``max_position_embeddings`` is set, the full cos/sin table is built eagerly
at construction time and registered as a buffer.  This avoids runtime
``Tensor.item()`` calls that would cause torch.compile graph breaks.

References:
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from einops import rearrange
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
            If set, the full cos/sin table is eagerly built at construction, avoiding
            graph breaks under torch.compile.
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
        if config.max_position_embeddings is not None and config.max_position_embeddings <= 0:
            raise ValueError(
                "RotaryConfig.max_position_embeddings must be > 0 when provided, "
                f"got {config.max_position_embeddings}."
            )

        self.config = config

        inv_freq = 1.0 / (
            config.base_theta
            ** (torch.arange(0, config.dim, 2, dtype=torch.float32) / float(config.dim))
        )
        # (dim/2,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # When max_position_embeddings is known, eagerly build the full table as
        # a buffer.  Buffers travel with .to(device) so no runtime rebuild needed.
        if config.max_position_embeddings is not None and config.max_position_embeddings > 0:
            t = torch.arange(config.max_position_embeddings, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            self.register_buffer("_cos_cached", freqs.cos(), persistent=False)
            self.register_buffer("_sin_cached", freqs.sin(), persistent=False)
        else:
            self.register_buffer(
                "_cos_cached", torch.empty(0, dtype=torch.float32), persistent=False
            )
            self.register_buffer(
                "_sin_cached", torch.empty(0, dtype=torch.float32), persistent=False
            )

    @torch.no_grad()
    def _grow_cache(self, max_pos: int, device: torch.device) -> None:
        """Grow the cos/sin cache to cover [0, max_pos) on *device*.

        Only used when ``max_position_embeddings`` is unset (dynamic mode).
        """
        if max_pos <= 0:
            return
        if self._cos_cached.shape[0] >= max_pos and self._cos_cached.device == device:
            return
        t = torch.arange(max_pos, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device=device))
        self._cos_cached = freqs.cos()
        self._sin_cached = freqs.sin()

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

        if position_ids.device != device:
            position_ids = position_ids.to(device)
        if position_ids.dtype not in (torch.int32, torch.int64):
            position_ids = position_ids.to(torch.int64)

        if position_ids.numel() == 0:
            b, seq = position_ids.shape
            empty = torch.empty((b, seq, self.config.dim // 2), device=device, dtype=dtype)
            return empty, empty

        is_compiling = bool(
            hasattr(torch, "_dynamo")
            and hasattr(torch._dynamo, "is_compiling")
            and torch._dynamo.is_compiling()
        )

        if (not is_compiling) and torch.any(position_ids < 0):
            raise ValueError("position_ids must be non-negative for RoPE.")

        # Dynamic mode: grow cache on demand (causes graph break under torch.compile).
        if self.config.max_position_embeddings is None:
            max_pos = int(position_ids.max().item()) + 1
            self._grow_cache(max_pos, device)
        elif not is_compiling:
            max_pos = int(position_ids.max().item()) + 1
            if max_pos > self._cos_cached.shape[0]:
                raise ValueError(
                    "position_ids exceed RotaryConfig.max_position_embeddings. "
                    f"max seen position={max_pos - 1}, cache size={self._cos_cached.shape[0]}."
                )

        # Gather from cache.  When max_position_embeddings is set, the cache is
        # a buffer that already lives on `device` (moved by .to(device)).
        b, seq = position_ids.shape
        flat = rearrange(position_ids, "b l -> (b l)")
        cos = rearrange(self._cos_cached.index_select(0, flat), "(b l) d -> b l d", b=b, l=seq).to(
            dtype=dtype
        )
        sin = rearrange(self._sin_cached.index_select(0, flat), "(b l) d -> b l d", b=b, l=seq).to(
            dtype=dtype
        )
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

        cos, sin = self._gather_cos_sin(position_ids, device=x.device, dtype=x.dtype)
        cos = rearrange(cos, "b l d -> b 1 l d")
        sin = rearrange(sin, "b l d -> b 1 l d")

        x_rope = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]

        # (B, H, L, dim) -> (B, H, L, dim/2, 2)
        x_rope = rearrange(x_rope, "b h l (d r) -> b h l d r", r=2)
        x1 = x_rope[..., 0]
        x2 = x_rope[..., 1]

        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        out = rearrange(torch.stack((out1, out2), dim=-1), "b h l d r -> b h l (d r)")

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
