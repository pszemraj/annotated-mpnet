"""Fast MPNet two-stream attention using RoPE + FlexAttention.

Why this exists
---------------
MPNet pretraining uses *two-stream attention* (content stream + query stream) with a
non-standard masking pattern (PLM-style triangular pieces + an extra block for MPNet).
Naively expressing this with dense masks/biases forces attention kernels to fall back,
and for long contexts it materializes O(L^2) boolean masks / bias tensors.

PyTorch FlexAttention can express this pattern as a structural BlockMask and run it
with a Flash-like kernel.

This module provides:
- Structural BlockMask builders for MPNet two-stream attention (cached).
- A RoPE + FlexAttention implementation of the two-stream self-attention.
- A RoPE + SDPA fallback (still correct, usually slower; also supports attention dropout).

IMPORTANT LIMITATION (PyTorch upstream)
-------------------------------------
FlexAttention currently has no built-in attention-dropout path (feature request is open). If
attention dropout is nonzero, we fall back to SDPA to keep training semantics correct.

As of PyTorch 2.10 docs, FlexAttention supports kernel tuning via kernel_options and a BACKEND
option (AUTO/TRITON/TRITON_DECODE/FLASH). Older PyTorch versions may not support all options; by
default we do NOT pass any kernel_options.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange

from .rotary_embedding import RotaryEmbedding


# FlexAttention is available in PyTorch >= 2.5 (prototype).
try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

    _FLEX_AVAILABLE = True
except Exception:  # pragma: no cover
    BlockMask = Any  # type: ignore
    create_block_mask = None  # type: ignore
    flex_attention = None  # type: ignore
    _FLEX_AVAILABLE = False

MaskModFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
ScoreModFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]


@dataclass(frozen=True)
class MPNetTwoStreamAttentionConfig:
    """Configuration for RoPE + FlexAttention two-stream attention."""

    # Main enable flag (still gated on FlexAttention availability + dropout==0)
    use_flex_attention: bool = True

    # Block size for create_block_mask (int or (BLOCK_M, BLOCK_N)).
    flex_block_size: Union[int, Tuple[int, int]] = 128

    # Whether to compile block-mask creation. Usually False; mask creation is cached anyway.
    flex_compile_block_mask: bool = False

    # Optional raw kernel options dict forwarded to flex_attention(..., kernel_options=...).
    # Example (PyTorch >= 2.10):
    #   {"BLOCK_M": 64, "BLOCK_N": 64, "PRESCALE_QK": True, "BACKEND": "TRITON"}
    # NOTE: Unsupported keys will raise at runtime; leave None unless you know what you're doing.
    flex_kernel_options: Optional[Dict[str, Any]] = None


def _dynamo_is_compiling() -> bool:
    """Return True when torch._dynamo is tracing/compiling."""
    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "is_compiling"):
        return torch._dynamo.is_compiling()
    return False


def _disable_compile(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that disables torch.compile/torch._dynamo for helper code."""
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
        return torch.compiler.disable(fn)  # type: ignore[attr-defined]
    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "disable"):
        return torch._dynamo.disable(fn)  # type: ignore[attr-defined]
    return fn


# ---- BlockMask cache ------------------------------------------------------

_BLOCK_MASK_CACHE_MAXSIZE = 64
# key: (seq_len, pred_size, device_type, device_index, block_size, compile_flag)
_BLOCK_MASK_CACHE: "OrderedDict[tuple[int, int, str, int, Union[int, Tuple[int,int]], bool], Tuple[BlockMask, BlockMask]]" = OrderedDict()


def _device_key(device: torch.device) -> Tuple[str, int]:
    """Build a stable cache key fragment from a torch device.

    :param torch.device device: Device to encode.
    :return Tuple[str, int]: ``(device_type, device_index)`` with index ``-1`` when absent.
    """
    idx = -1 if device.index is None else int(device.index)
    return device.type, idx


def _mpnet_query_mask_mod(seq_len: int, pred_size: int) -> MaskModFn:
    """Return mask_mod(b,h,q_idx,kv_idx)->bool for the *query* stream.

    True means the attention edge is **allowed**.
    """
    seq_left = seq_len - pred_size

    def mask_mod(
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Return allowed query-stream edges for MPNet two-stream attention.

        :param torch.Tensor b: Batch index tensor (unused, required by FlexAttention API).
        :param torch.Tensor h: Head index tensor (unused, required by FlexAttention API).
        :param torch.Tensor q_idx: Query positions.
        :param torch.Tensor kv_idx: Key/value positions.
        :return torch.Tensor: Boolean tensor where ``True`` marks allowed attention edges.
        """
        # 1) allow all original (non-predicted) tokens
        allowed = kv_idx < seq_left

        # 2) in predicted-token block, allow strictly earlier predicted positions
        kv_pred = (kv_idx >= seq_left) & (kv_idx < seq_len)
        allowed = allowed | (kv_pred & ((kv_idx - seq_left) < q_idx))

        # 3) in mask-token block, allow positions at/after the query index (MPNet extra block)
        kv_mask = kv_idx >= seq_len
        allowed = allowed | (kv_mask & ((kv_idx - seq_len) >= q_idx))
        return allowed

    return mask_mod


def _mpnet_content_mask_mod(seq_len: int, pred_size: int) -> MaskModFn:
    """Return mask_mod(b,h,q_idx,kv_idx)->bool for the *content* stream.

    True means the attention edge is **allowed**.
    """
    seq_left = seq_len - pred_size

    def mask_mod(
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Return allowed content-stream edges for MPNet two-stream attention.

        :param torch.Tensor b: Batch index tensor (unused, required by FlexAttention API).
        :param torch.Tensor h: Head index tensor (unused, required by FlexAttention API).
        :param torch.Tensor q_idx: Query positions.
        :param torch.Tensor kv_idx: Key/value positions.
        :return torch.Tensor: Boolean tensor where ``True`` marks allowed attention edges.
        """
        pred_q = (q_idx >= seq_left) & (q_idx < seq_len)

        # 1) allow all original (non-predicted) tokens
        allowed = kv_idx < seq_left

        # 2) predicted-token keys are only visible to predicted-token queries in lower-tri pattern
        kv_pred = (kv_idx >= seq_left) & (kv_idx < seq_len)
        allowed = allowed | (kv_pred & pred_q & ((kv_idx - seq_left) <= (q_idx - seq_left)))

        # 3) mask-token keys are visible to:
        #   - all non-predicted queries
        #   - predicted queries but only for strictly future mask positions
        kv_mask = kv_idx >= seq_len
        allowed = allowed | (kv_mask & (~pred_q | ((kv_idx - seq_len) > (q_idx - seq_left))))
        return allowed

    return mask_mod


@_disable_compile
def get_mpnet_two_stream_block_masks(
    *,
    seq_len: int,
    pred_size: int,
    device: torch.device,
    block_size: Union[int, Tuple[int, int]] = 128,
    compile_block_mask: bool = False,
) -> Tuple[BlockMask, BlockMask]:
    """Return (content_block_mask, query_block_mask) for MPNet two-stream attention.

    seq_len is the original sequence length (before appending mask symbols twice).

    This uses a small manual LRU cache keyed by shape/device. Caching is bypassed while
    torch.compile is tracing to avoid graph breaks.
    """
    if pred_size <= 0 or pred_size > seq_len:
        raise ValueError(
            f"pred_size must be in [1, seq_len]. Got pred_size={pred_size}, seq_len={seq_len}."
        )

    if not _FLEX_AVAILABLE:
        raise RuntimeError("FlexAttention is not available in this PyTorch build.")

    device_type, device_index = _device_key(device)
    key = (seq_len, pred_size, device_type, device_index, block_size, bool(compile_block_mask))

    if _dynamo_is_compiling():
        return _build_mpnet_two_stream_block_masks(
            seq_len=seq_len,
            pred_size=pred_size,
            device=device,
            block_size=block_size,
            compile_block_mask=compile_block_mask,
        )

    cached = _BLOCK_MASK_CACHE.get(key)
    if cached is not None:
        _BLOCK_MASK_CACHE.move_to_end(key)
        return cached

    masks = _build_mpnet_two_stream_block_masks(
        seq_len=seq_len,
        pred_size=pred_size,
        device=device,
        block_size=block_size,
        compile_block_mask=compile_block_mask,
    )

    _BLOCK_MASK_CACHE[key] = masks
    _BLOCK_MASK_CACHE.move_to_end(key)
    if len(_BLOCK_MASK_CACHE) > _BLOCK_MASK_CACHE_MAXSIZE:
        _BLOCK_MASK_CACHE.popitem(last=False)

    return masks


def _build_mpnet_two_stream_block_masks(
    *,
    seq_len: int,
    pred_size: int,
    device: torch.device,
    block_size: Union[int, Tuple[int, int]] = 128,
    compile_block_mask: bool = False,
) -> Tuple[BlockMask, BlockMask]:
    """Build (content_block_mask, query_block_mask) without caching."""
    kv_len = seq_len + pred_size

    query_mask_mod = _mpnet_query_mask_mod(seq_len=seq_len, pred_size=pred_size)
    content_mask_mod = _mpnet_content_mask_mod(seq_len=seq_len, pred_size=pred_size)

    # create_block_mask expects device as a string (e.g. "cuda:0", "cpu").
    device_str = str(device)

    query_bm = create_block_mask(
        query_mask_mod,
        B=None,
        H=None,
        Q_LEN=pred_size,
        KV_LEN=kv_len,
        device=device_str,
        BLOCK_SIZE=block_size,
        _compile=compile_block_mask,
    )

    content_bm = create_block_mask(
        content_mask_mod,
        B=None,
        H=None,
        Q_LEN=kv_len,
        KV_LEN=kv_len,
        device=device_str,
        BLOCK_SIZE=block_size,
        _compile=compile_block_mask,
    )

    return content_bm, query_bm


# ---- Optional key padding (score_mod) ------------------------------------


def make_key_padding_score_mod(key_padding_mask: torch.Tensor) -> ScoreModFn:
    """Create a score_mod closure that masks out padded keys.

    Args:
        key_padding_mask: Bool tensor shaped (B, KV_LEN) with True indicating padding.

    Returns:
        score_mod callable compatible with flex_attention.
    """
    if key_padding_mask.dtype != torch.bool:
        key_padding_mask = key_padding_mask.to(torch.bool)

    def score_mod(
        score: torch.Tensor,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Mask padded keys by writing ``-inf`` to matching attention scores.

        :param torch.Tensor score: Current attention score tensor.
        :param torch.Tensor b: Batch index tensor.
        :param torch.Tensor h: Head index tensor (unused).
        :param torch.Tensor q_idx: Query positions (unused).
        :param torch.Tensor kv_idx: Key/value positions.
        :return torch.Tensor: Scores with padded key positions set to ``-inf``.
        """
        # key_padding_mask[b, kv_idx] broadcasts to score shape.
        return torch.where(key_padding_mask[b, kv_idx], float("-inf"), score)

    return score_mod


# ---- SDPA helpers ---------------------------------------------------------


def _build_two_stream_bool_masks(
    *,
    seq_len: int,
    pred_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (query_mask, content_mask) boolean masks matching make_query_and_content_mask.

    True means masked (disallowed). Shapes:
        query_mask   : (pred_size, key_len)
        content_mask : (key_len, key_len)
    where key_len = seq_len + pred_size.
    """
    seq_left = seq_len - pred_size
    key_len = seq_len + pred_size

    # Query mask (bool) - True means masked.
    tri_upper = torch.triu(
        torch.ones(pred_size, pred_size, device=device, dtype=torch.bool), diagonal=0
    )
    left_block = torch.zeros(pred_size, seq_left, device=device, dtype=torch.bool)
    query_mask = torch.cat((left_block, tri_upper, ~tri_upper), dim=-1)
    assert query_mask.shape == (pred_size, key_len)

    # Content mask (bool) - True means masked.
    top = torch.zeros(seq_left, pred_size, device=device, dtype=torch.bool)
    tri_lower = torch.tril(
        torch.ones(pred_size, pred_size, device=device, dtype=torch.bool), diagonal=0
    )
    bottom = torch.zeros(pred_size, pred_size, device=device, dtype=torch.bool)
    base = torch.cat((top, tri_lower, bottom), dim=0)  # (key_len, pred_size)
    left_block2 = torch.zeros(key_len, seq_left, device=device, dtype=torch.bool)
    content_mask = torch.cat((left_block2, ~base, base), dim=-1)
    assert content_mask.shape == (key_len, key_len)

    return query_mask, content_mask


def _build_sdpa_attn_bias(
    *,
    attn_mask: Optional[torch.Tensor],
    key_padding_mask: Optional[torch.Tensor],
    bsz: int,
    num_heads: int,
    tgt_len: int,
    src_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Build an additive attention bias tensor for SDPA.

    Returns a tensor shaped (B, H, tgt, src) or None.
    """
    if attn_mask is None and key_padding_mask is None:
        return None

    attn_bias = torch.zeros((bsz, num_heads, tgt_len, src_len), device=device, dtype=dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            mask = attn_mask
            if mask.dim() == 2:
                mask = rearrange(mask, "t s -> 1 1 t s")
            elif mask.dim() == 3:
                mask = rearrange(mask, "b t s -> b 1 t s")
            attn_bias.masked_fill_(mask.to(device=device), float("-inf"))
        else:
            mask = attn_mask.to(device=device, dtype=dtype)
            if mask.dim() == 2:
                mask = rearrange(mask, "t s -> 1 1 t s")
            elif mask.dim() == 3:
                mask = rearrange(mask, "b t s -> b 1 t s")
            attn_bias += mask

    if key_padding_mask is not None:
        key_mask = rearrange(key_padding_mask.to(torch.bool), "b s -> b 1 1 s")
        attn_bias.masked_fill_(key_mask, float("-inf"))

    return attn_bias


# ---- Main attention implementations --------------------------------------


def two_stream_self_attention_rope_flex(
    attn: Any,
    *,
    c: torch.Tensor,
    q: torch.Tensor,
    content_positions: torch.Tensor,
    query_positions: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor],
    rope: RotaryEmbedding,
    config: MPNetTwoStreamAttentionConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute two-stream attention with RoPE + FlexAttention.

    Args:
        attn: RelativeMultiHeadAttention instance (used for projections/out_proj).
        c: Content stream (T_c, B, E) where T_c = seq_len + pred_size.
        q: Query stream (pred_size, B, E).
        content_positions: (B, T_c) permuted position ids for content stream tokens.
        query_positions: (B, pred_size) permuted position ids for query stream tokens.
        key_padding_mask: Optional (B, T_c) boolean mask for keys.
        rope: RotaryEmbedding.
        config: MPNetTwoStreamAttentionConfig.

    Returns:
        (c_out, q_out) in the same shapes as (c, q).
    """
    if not _FLEX_AVAILABLE:
        raise RuntimeError("FlexAttention is not available; cannot run rope+flex path.")

    # FlexAttention has no attention-dropout; enforce semantics.
    if float(getattr(attn, "dropout", 0.0)) != 0.0:
        raise ValueError(
            "RoPE+FlexAttention requires attention dropout == 0.0 for semantic equivalence. "
            f"Got attention dropout={getattr(attn, 'dropout', None)}."
        )

    t_c, bsz, embed_dim = c.shape
    t_q = q.shape[0]
    pred_size = t_q
    seq_len = t_c - pred_size
    if seq_len <= 0:
        raise ValueError(f"Invalid two-stream shapes: content_len={t_c}, pred_size={pred_size}.")

    if content_positions.shape != (bsz, t_c):
        raise ValueError(
            f"content_positions must have shape (B, {t_c}); got {tuple(content_positions.shape)}."
        )
    if query_positions.shape != (bsz, t_q):
        raise ValueError(
            f"query_positions must have shape (B, {t_q}); got {tuple(query_positions.shape)}."
        )

    # Projections (T,B,E)
    q_c = attn.in_proj_q(c)
    q_q = attn.in_proj_q(q)
    k = attn.in_proj_k(c)
    v = attn.in_proj_v(c)

    # (T,B,E) -> (B,H,T,D)
    num_heads = attn.num_heads
    head_dim = embed_dim // num_heads
    if head_dim * num_heads != embed_dim:
        raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}.")

    q_c = rearrange(q_c, "t b (h d) -> b h t d", h=num_heads).contiguous()
    q_q = rearrange(q_q, "t b (h d) -> b h t d", h=num_heads).contiguous()
    k = rearrange(k, "t b (h d) -> b h t d", h=num_heads).contiguous()
    v = rearrange(v, "t b (h d) -> b h t d", h=num_heads).contiguous()

    # RoPE (positions are permuted ids)
    content_positions = content_positions.to(dtype=torch.long, device=c.device)
    query_positions = query_positions.to(dtype=torch.long, device=c.device)

    k = rope.rotate(k, content_positions)
    q_c = rope.rotate(q_c, content_positions)
    q_q = rope.rotate(q_q, query_positions)

    # Structural BlockMasks (cached)
    content_bm, query_bm = get_mpnet_two_stream_block_masks(
        seq_len=seq_len,
        pred_size=pred_size,
        device=c.device,
        block_size=config.flex_block_size,
        compile_block_mask=config.flex_compile_block_mask,
    )

    score_mod = None
    if key_padding_mask is not None:
        if key_padding_mask.shape != (bsz, t_c):
            raise ValueError(
                f"key_padding_mask must have shape (B, {t_c}); got {tuple(key_padding_mask.shape)}."
            )
        score_mod = make_key_padding_score_mod(
            key_padding_mask.to(device=c.device, dtype=torch.bool)
        )

    # Optional kernel tuning
    kernel_options: Optional[Dict[str, Any]] = None
    if config.flex_kernel_options is not None:
        kernel_options = dict(config.flex_kernel_options)

    scale = getattr(attn, "scaling", None)

    out_c = flex_attention(
        q_c,
        k,
        v,
        score_mod=score_mod,
        block_mask=content_bm,
        scale=scale,
        kernel_options=kernel_options,
    )
    out_q = flex_attention(
        q_q,
        k,
        v,
        score_mod=score_mod,
        block_mask=query_bm,
        scale=scale,
        kernel_options=kernel_options,
    )

    # (B,H,T,D) -> (T,B,E)
    out_c = rearrange(out_c, "b h t d -> t b (h d)")
    out_q = rearrange(out_q, "b h t d -> t b (h d)")

    return attn.out_proj(out_c), attn.out_proj(out_q)


def two_stream_self_attention_rope_sdpa(
    attn: Any,
    *,
    c: torch.Tensor,
    q: torch.Tensor,
    content_mask: Optional[torch.Tensor],
    query_mask: Optional[torch.Tensor],
    content_positions: torch.Tensor,
    query_positions: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor],
    rope: RotaryEmbedding,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RoPE + SDPA two-stream attention (correct, but often slower than flex).

    Supports attention dropout via SDPA.
    """
    if not hasattr(F, "scaled_dot_product_attention"):
        raise RuntimeError("scaled_dot_product_attention is not available in this PyTorch build.")

    t_c, bsz, embed_dim = c.shape
    t_q = q.shape[0]
    pred_size = t_q
    seq_len = t_c - pred_size
    if seq_len <= 0:
        raise ValueError(f"Invalid two-stream shapes: content_len={t_c}, pred_size={pred_size}.")

    num_heads = attn.num_heads
    head_dim = embed_dim // num_heads
    if head_dim * num_heads != embed_dim:
        raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}.")

    # If masks weren't provided, build them locally (depends only on seq_len/pred_size).
    if query_mask is None or content_mask is None:
        q_mask, c_mask = _build_two_stream_bool_masks(
            seq_len=seq_len, pred_size=pred_size, device=c.device
        )
        query_mask = q_mask if query_mask is None else query_mask
        content_mask = c_mask if content_mask is None else content_mask

    # Projections
    q_c = attn.in_proj_q(c)
    q_q = attn.in_proj_q(q)
    k = attn.in_proj_k(c)
    v = attn.in_proj_v(c)

    # (T,B,E) -> (B,H,T,D)
    q_c = rearrange(q_c, "t b (h d) -> b h t d", h=num_heads).contiguous()
    q_q = rearrange(q_q, "t b (h d) -> b h t d", h=num_heads).contiguous()
    k = rearrange(k, "t b (h d) -> b h t d", h=num_heads).contiguous()
    v = rearrange(v, "t b (h d) -> b h t d", h=num_heads).contiguous()

    # RoPE
    content_positions = content_positions.to(dtype=torch.long, device=c.device)
    query_positions = query_positions.to(dtype=torch.long, device=c.device)

    k = rope.rotate(k, content_positions)
    q_c = rope.rotate(q_c, content_positions)
    q_q = rope.rotate(q_q, query_positions)

    tgt_len_c = q_c.size(2)
    tgt_len_q = q_q.size(2)
    src_len = k.size(2)

    attn_bias_c = _build_sdpa_attn_bias(
        attn_mask=content_mask,
        key_padding_mask=key_padding_mask,
        bsz=bsz,
        num_heads=num_heads,
        tgt_len=tgt_len_c,
        src_len=src_len,
        device=q_c.device,
        dtype=q_c.dtype,
    )
    attn_bias_q = _build_sdpa_attn_bias(
        attn_mask=query_mask,
        key_padding_mask=key_padding_mask,
        bsz=bsz,
        num_heads=num_heads,
        tgt_len=tgt_len_q,
        src_len=src_len,
        device=q_q.device,
        dtype=q_q.dtype,
    )

    dropout_p = float(getattr(attn, "dropout", 0.0)) if getattr(attn, "training", False) else 0.0

    out_c = F.scaled_dot_product_attention(
        q_c,
        k,
        v,
        attn_mask=attn_bias_c,
        dropout_p=dropout_p,
        scale=None,
    )
    out_q = F.scaled_dot_product_attention(
        q_q,
        k,
        v,
        attn_mask=attn_bias_q,
        dropout_p=dropout_p,
        scale=None,
    )

    out_c = rearrange(out_c, "b h t d -> t b (h d)")
    out_q = rearrange(out_q, "b h t d -> t b (h d)")

    return attn.out_proj(out_c), attn.out_proj(out_q)


def two_stream_self_attention_rope(
    attn: Any,
    *,
    c: torch.Tensor,
    q: torch.Tensor,
    content_mask: Optional[torch.Tensor],
    query_mask: Optional[torch.Tensor],
    content_positions: torch.Tensor,
    query_positions: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor],
    rope: RotaryEmbedding,
    config: MPNetTwoStreamAttentionConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dispatcher: prefer RoPE+FlexAttention; fall back to RoPE+SDPA.

    Flex is used only when:
    - config.use_flex_attention == True
    - FlexAttention is available in this PyTorch build
    - attention dropout == 0.0 (FlexAttention limitation)

    For Flex path, content_mask/query_mask are ignored and may be None.
    """
    use_flex = (
        bool(config.use_flex_attention)
        and _FLEX_AVAILABLE
        and float(getattr(attn, "dropout", 0.0)) == 0.0
    )

    if use_flex:
        return two_stream_self_attention_rope_flex(
            attn,
            c=c,
            q=q,
            content_positions=content_positions,
            query_positions=query_positions,
            key_padding_mask=key_padding_mask,
            rope=rope,
            config=config,
        )

    return two_stream_self_attention_rope_sdpa(
        attn,
        c=c,
        q=q,
        content_mask=content_mask,
        query_mask=query_mask,
        content_positions=content_positions,
        query_positions=query_positions,
        key_padding_mask=key_padding_mask,
        rope=rope,
    )
