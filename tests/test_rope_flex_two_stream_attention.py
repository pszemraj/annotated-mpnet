"""Deterministic unit tests for RoPE + FlexAttention MPNet two-stream attention.

Drop this file into the repo as:
    tests/test_rope_flex_two_stream_attention.py

What this covers
----------------
1) RoPE correctness for *arbitrary* position ids (MPNet uses permuted positions).
2) Two-stream mask correctness: FlexAttention mask_mod closures match the existing dense
   query/content masks produced by make_query_and_content_mask.
3) Numerical equivalence: RoPE+FlexAttention matches RoPE+SDPA when attention_dropout == 0.

These tests are small and CPU-safe. FlexAttention is optional; if unavailable,
Flex-specific tests are skipped.
"""

from __future__ import annotations

import unittest

import torch

from annotated_mpnet.modeling.mpnet_for_pretraining import make_query_and_content_mask
from annotated_mpnet.transformer_modules import RelativeMultiHeadAttention
from annotated_mpnet.transformer_modules.rotary_embedding import RotaryConfig, RotaryEmbedding
from annotated_mpnet.transformer_modules.mpnet_flex_rope_attention import (
    MPNetTwoStreamAttentionConfig,
    _FLEX_AVAILABLE,
    _mpnet_content_mask_mod,
    _mpnet_query_mask_mod,
    two_stream_self_attention_rope_flex,
    two_stream_self_attention_rope_sdpa,
)


class TestRotaryEmbedding(unittest.TestCase):
    def test_rope_matches_reference_for_arbitrary_positions(self) -> None:
        torch.manual_seed(0)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        dtype = torch.float32

        B, H, L, D = 2, 3, 5, 8
        x = torch.randn(B, H, L, D, device=device, dtype=dtype)

        # Arbitrary (non-monotonic) positions.
        position_ids = torch.tensor(
            [[0, 4, 1, 3, 2], [7, 2, 5, 0, 6]], device=device, dtype=torch.long
        )

        rope = RotaryEmbedding(
            RotaryConfig(dim=D, base_theta=10_000.0, max_position_embeddings=128)
        ).to(device)
        y = rope.rotate(x, position_ids)

        # Reference implementation (explicit).
        inv_freq = rope.inv_freq.to(device=device, dtype=torch.float32)  # (D/2,)
        freqs = position_ids.to(torch.float32)[..., None] * inv_freq[None, None, :]  # (B,L,D/2)
        cos = freqs.cos().to(dtype=dtype)
        sin = freqs.sin().to(dtype=dtype)

        x_rope = x.view(B, H, L, D // 2, 2)
        x_even = x_rope[..., 0]
        x_odd = x_rope[..., 1]

        cos = cos[:, None, :, :]  # (B,1,L,D/2)
        sin = sin[:, None, :, :]

        y_even = x_even * cos - x_odd * sin
        y_odd = x_even * sin + x_odd * cos
        y_ref = torch.stack((y_even, y_odd), dim=-1).reshape(B, H, L, D)

        torch.testing.assert_close(y, y_ref, rtol=0.0, atol=1e-6)


class TestTwoStreamMaskMods(unittest.TestCase):
    def test_mask_mods_match_dense_two_stream_masks(self) -> None:
        """Verify the FlexAttention mask_mod closures match the existing dense boolean masks.

        Dense masks in mpnet_for_pretraining.py use True = masked.
        FlexAttention mask_mod uses True = allowed.
        So we check: allowed == ~dense_mask.
        """

        torch.manual_seed(0)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # A small grid of sizes + an edge case.
        cases = [
            (8, 1),
            (8, 3),
            (16, 5),
            (32, 8),
        ]

        for seq_len, pred_size in cases:
            key_len = seq_len + pred_size
            dummy_input_ids = torch.zeros(2, key_len, device=device, dtype=torch.long)

            query_mask, content_mask, _ = make_query_and_content_mask(
                dummy_input_ids, seq_len, pred_size, pad_token_id=None, attention_mask=None
            )
            # query_mask:   (pred_size, key_len), True=masked
            # content_mask: (key_len, key_len), True=masked

            # Build allowed masks from closures.
            q_mod = _mpnet_query_mask_mod(seq_len=seq_len, pred_size=pred_size)
            c_mod = _mpnet_content_mask_mod(seq_len=seq_len, pred_size=pred_size)

            q_idx = torch.arange(pred_size, device=device)[:, None]
            kv_idx = torch.arange(key_len, device=device)[None, :]
            allowed_q = q_mod(
                torch.tensor(0, device=device),
                torch.tensor(0, device=device),
                q_idx,
                kv_idx,
            )

            q_idx2 = torch.arange(key_len, device=device)[:, None]
            kv_idx2 = torch.arange(key_len, device=device)[None, :]
            allowed_c = c_mod(
                torch.tensor(0, device=device),
                torch.tensor(0, device=device),
                q_idx2,
                kv_idx2,
            )

            self.assertTrue(torch.equal(allowed_q.to(torch.bool), (~query_mask).to(torch.bool)))
            self.assertTrue(torch.equal(allowed_c.to(torch.bool), (~content_mask).to(torch.bool)))


class TestTwoStreamAttentionParity(unittest.TestCase):
    def test_rope_flex_matches_rope_sdpa_dropout0(self) -> None:
        """FlexAttention output should match SDPA when attention_dropout == 0.

        This is the critical correctness gate before you trust perf numbers.
        """

        if not _FLEX_AVAILABLE:
            self.skipTest("FlexAttention not available in this PyTorch build")

        torch.manual_seed(0)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        dtype = torch.float32

        B = 2
        seq_len = 16
        pred_size = 5
        key_len = seq_len + pred_size

        embed_dim = 32
        num_heads = 4
        head_dim = embed_dim // num_heads

        # Attention module with dropout=0.
        attn = RelativeMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            self_attention=True,
        ).to(device=device, dtype=dtype)
        attn.train(True)

        # Inputs.
        c = torch.randn(key_len, B, embed_dim, device=device, dtype=dtype)
        q = torch.randn(pred_size, B, embed_dim, device=device, dtype=dtype)

        # MPNet-style permuted positions: [perm(seq_len), mask_pos, mask_pos]
        perm = torch.stack([torch.randperm(seq_len, device=device) for _ in range(B)], dim=0)
        mask_pos = perm[:, -pred_size:]
        positions = torch.cat([perm, mask_pos, mask_pos], dim=1)

        content_positions = positions[:, :key_len]
        query_positions = positions[:, -pred_size:]

        # Dense two-stream masks (for SDPA reference).
        dummy_input_ids = torch.zeros(B, key_len, device=device, dtype=torch.long)
        query_mask, content_mask, _ = make_query_and_content_mask(
            dummy_input_ids, seq_len, pred_size, pad_token_id=None, attention_mask=None
        )

        # Key padding mask: mask a couple of keys.
        key_padding_mask = torch.zeros(B, key_len, device=device, dtype=torch.bool)
        key_padding_mask[0, 0] = True
        key_padding_mask[1, 3] = True

        rope = RotaryEmbedding(
            RotaryConfig(dim=head_dim, base_theta=10_000.0, max_position_embeddings=2048)
        ).to(device)

        cfg = MPNetTwoStreamAttentionConfig(
            use_flex_attention=True,
            flex_block_size=128,
            flex_compile_block_mask=False,
            flex_kernel_options=None,
        )

        c_flex, q_flex = two_stream_self_attention_rope_flex(
            attn,
            c=c,
            q=q,
            content_positions=content_positions,
            query_positions=query_positions,
            key_padding_mask=key_padding_mask,
            rope=rope,
            config=cfg,
        )

        c_sdpa, q_sdpa = two_stream_self_attention_rope_sdpa(
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

        # Unfused FlexAttention (without torch.compile) and SDPA use different
        # numerical paths; allow slightly larger tolerance for CPU / unfused runs.
        torch.testing.assert_close(c_flex, c_sdpa, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(q_flex, q_sdpa, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
