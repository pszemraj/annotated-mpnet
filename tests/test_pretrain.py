"""Unit tests for pretraining helper utilities."""

import json
from pathlib import Path
import sys
import unittest
from argparse import Namespace
from tempfile import TemporaryDirectory

import torch
import torch.nn.functional as F

from annotated_mpnet.modeling.mpnet_for_pretraining import (
    init_final_params,
    make_query_and_content_mask,
    two_stream_self_attention,
)
from annotated_mpnet.scheduler import PolynomialDecayLRScheduler
from annotated_mpnet.transformer_modules import RelativeMultiHeadAttention, SentenceEncoder
from annotated_mpnet.utils import utils

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cli_tools import pretrain_mpnet
from tests.dummy_tokenizer import DummyTokenizer


class TestPretrainHelpers(unittest.TestCase):
    """Unit tests for pretrain helper functions."""

    def test_get_initial_best_loss(self) -> None:
        """Ensure best-loss initialization falls back correctly.

        :return None: This test returns nothing.
        """
        self.assertEqual(
            pretrain_mpnet._get_initial_best_loss(None), pretrain_mpnet.DEFAULT_BEST_LOSS
        )
        self.assertEqual(pretrain_mpnet._get_initial_best_loss({"best_loss": 1.23}), 1.23)
        self.assertEqual(
            pretrain_mpnet._get_initial_best_loss({"steps": 5}),
            pretrain_mpnet.DEFAULT_BEST_LOSS,
        )

    def test_validate_tokenizer_vocab_size_matches(self) -> None:
        """Ensure tokenizer vocab size matches checkpoint size.

        :return None: This test returns nothing.
        """
        args = Namespace(original_vocab_size=10, padded_vocab_size=16)

        class DummyTokenizer:
            def __len__(self) -> int:
                return 10

        pretrain_mpnet._validate_tokenizer_vocab_size(DummyTokenizer(), args, "checkpoint")

    def test_validate_tokenizer_vocab_size_mismatch_raises(self) -> None:
        """Raise when tokenizer vocab size mismatches checkpoint size.

        :return None: This test returns nothing.
        """
        args = Namespace(original_vocab_size=10, padded_vocab_size=16)

        class DummyTokenizer:
            def __len__(self) -> int:
                return 11

        with self.assertRaises(ValueError):
            pretrain_mpnet._validate_tokenizer_vocab_size(DummyTokenizer(), args, "checkpoint")

    def test_weight_decay_grouping(self) -> None:
        """Ensure biases and norm weights are excluded from weight decay.

        :return None: This test returns nothing.
        """

        class DummyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.norm = torch.nn.LayerNorm(4)

        model = DummyModel()
        decay, no_decay = pretrain_mpnet._group_parameters_for_weight_decay(model)

        self.assertTrue(any(p is model.linear.weight for p in decay))
        self.assertTrue(any(p is model.linear.bias for p in no_decay))
        self.assertTrue(any(p is model.norm.weight for p in no_decay))
        self.assertTrue(any(p is model.norm.bias for p in no_decay))

    def test_cli_flag_was_provided(self) -> None:
        """Detect CLI flags provided as separate arg or equals form.

        :return None: This test returns nothing.
        """
        argv = ["--use-rope", "--attention-dropout=0.0"]
        self.assertTrue(pretrain_mpnet._cli_flag_was_provided(argv, "--attention-dropout"))
        self.assertFalse(pretrain_mpnet._cli_flag_was_provided(argv, "--compile"))

    def test_normalize_attention_dropout_for_flex_sets_default(self) -> None:
        """Set attention_dropout to 0.0 for new RoPE+Flex runs when not explicit.

        :return None: This test returns nothing.
        """
        args = Namespace(use_rope=True, use_flex_attention=True, attention_dropout=0.1)
        pretrain_mpnet._normalize_attention_dropout_for_flex(
            args,
            arch_source="new",
            attention_dropout_explicit=False,
        )
        self.assertEqual(args.attention_dropout, 0.0)

    def test_normalize_attention_dropout_for_flex_respects_explicit(self) -> None:
        """Keep explicit attention_dropout values unchanged.

        :return None: This test returns nothing.
        """
        args = Namespace(use_rope=True, use_flex_attention=True, attention_dropout=0.1)
        pretrain_mpnet._normalize_attention_dropout_for_flex(
            args,
            arch_source="new",
            attention_dropout_explicit=True,
        )
        self.assertEqual(args.attention_dropout, 0.1)

    def test_polynomial_scheduler_step_indexing(self) -> None:
        """Ensure scheduler step indexing matches expected warmup/decay behavior.

        :return None: This test returns nothing.
        """
        args = Namespace(
            lr=1.0,
            warmup_updates=2,
            total_updates=4,
            end_learning_rate=0.0,
            power=1.0,
        )
        param = torch.nn.Parameter(torch.randn(()))
        optimizer = torch.optim.SGD([param], lr=0.0)
        scheduler = PolynomialDecayLRScheduler(args, optimizer)

        self.assertAlmostEqual(scheduler.get_lr(), 0.5, places=6)
        self.assertAlmostEqual(scheduler.step(1), 0.5, places=6)
        self.assertAlmostEqual(scheduler.step(2), 1.0, places=6)
        self.assertAlmostEqual(scheduler.step(3), 0.5, places=6)

        args_no_warmup = Namespace(
            lr=1.0,
            warmup_updates=0,
            total_updates=4,
            end_learning_rate=0.0,
            power=1.0,
        )
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.randn(()))], lr=0.0)
        scheduler = PolynomialDecayLRScheduler(args_no_warmup, optimizer)
        self.assertAlmostEqual(scheduler.step(1), 1.0, places=6)
        self.assertAlmostEqual(scheduler.step(4), 0.0, places=6)

    def test_init_final_params_zeroes_padding_idx(self) -> None:
        """Ensure padding_idx=0 embeddings are zeroed during init.

        :return None: This test returns nothing.
        """
        emb = torch.nn.Embedding(10, 4, padding_idx=0)
        emb.weight.data.uniform_(-1.0, 1.0)
        init_final_params(emb)
        self.assertTrue(torch.allclose(emb.weight.data[0], torch.zeros_like(emb.weight.data[0])))

    def test_encode_emb_handles_sinusoidal_positions(self) -> None:
        """Ensure encode_emb supports sinusoidal embeddings and masks padding.

        :return None: This test returns nothing.
        """
        model = SentenceEncoder(
            padding_idx=0,
            vocab_size=32,
            num_encoder_layers=1,
            embedding_dim=16,
            ffn_embedding_dim=32,
            num_attention_heads=2,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            max_seq_len=8,
            num_segments=0,
            encoder_normalize_before=True,
            activation_fn="gelu",
            normalize_before=False,
            learned_pos_embedding=False,
            relative_attention_num_buckets=8,
            relative_attention_max_distance=16,
        )
        input_ids = torch.tensor([[0, 5, 6, 0]])
        positions = torch.tensor([[0, 1, 2, 3]])
        emb = model.encode_emb(input_ids, positions=positions)
        self.assertEqual(emb.shape[:2], input_ids.shape)
        self.assertTrue(torch.allclose(emb[0, 0], torch.zeros_like(emb[0, 0])))
        self.assertTrue(torch.allclose(emb[0, 3], torch.zeros_like(emb[0, 3])))

    def test_hf_max_positions_to_internal(self) -> None:
        """Ensure HF max positions convert to internal max_positions.

        :return None: This test returns nothing.
        """
        self.assertEqual(utils.hf_max_positions_to_internal(514), 512)
        self.assertEqual(utils.hf_max_positions_to_internal(2), 1)

    def test_lm_head_weight_registered_and_tied(self) -> None:
        """Ensure lm_head.weight is registered and tied to embeddings.

        :return None: This test returns nothing.
        """
        args = Namespace(
            encoder_layers=2,
            encoder_embed_dim=64,
            encoder_ffn_dim=128,
            encoder_attention_heads=2,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.1,
            activation_fn="gelu",
            max_positions=32,
            relative_attention_num_buckets=16,
            relative_attention_max_distance=64,
            normalize_before=False,
            padded_vocab_size=30528,
        )
        tokenizer = DummyTokenizer()
        model = pretrain_mpnet.MPNetForPretraining(args, tokenizer)

        state_dict = model.state_dict()
        self.assertIn("lm_head.weight", state_dict)
        self.assertIn("sentence_encoder.embed_tokens.weight", state_dict)
        self.assertIs(model.lm_head.weight, model.sentence_encoder.embed_tokens.weight)

    def test_sentence_encoder_gradient_checkpointing(self) -> None:
        """Ensure gradient checkpointing path runs without error.

        :return None: This test returns nothing.
        """
        model = SentenceEncoder(
            padding_idx=1,
            vocab_size=128,
            num_encoder_layers=2,
            embedding_dim=32,
            ffn_embedding_dim=64,
            num_attention_heads=4,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.1,
            max_seq_len=16,
            num_segments=0,
            encoder_normalize_before=True,
            activation_fn="gelu",
            normalize_before=False,
            relative_attention_num_buckets=8,
            relative_attention_max_distance=16,
            gradient_checkpointing=True,
        )
        model.train()
        tokens = torch.randint(0, 128, (2, 8))
        inner_states, sentence_rep = model(tokens)
        self.assertEqual(inner_states[-1].shape[0], tokens.shape[0])
        self.assertEqual(inner_states[-1].shape[1], tokens.shape[1])
        self.assertEqual(sentence_rep.shape[0], tokens.shape[0])

    def test_sentence_encoder_inner_states_layout(self) -> None:
        """Ensure inner_states layout is consistent across last_state_only modes.

        :return None: This test returns nothing.
        """
        model = SentenceEncoder(
            padding_idx=1,
            vocab_size=128,
            num_encoder_layers=1,
            embedding_dim=32,
            ffn_embedding_dim=64,
            num_attention_heads=4,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.1,
            max_seq_len=16,
            num_segments=0,
            encoder_normalize_before=True,
            activation_fn="gelu",
            normalize_before=False,
            relative_attention_num_buckets=8,
            relative_attention_max_distance=16,
        )
        tokens = torch.randint(0, 128, (2, 8))
        all_states, _ = model(tokens, last_state_only=False)
        last_only, _ = model(tokens, last_state_only=True)
        self.assertEqual(all_states[-1].shape, last_only[0].shape)
        self.assertEqual(all_states[-1].shape[:2], tokens.shape)

    def test_sentence_encoder_positions_match_default(self) -> None:
        """Ensure explicit positions follow the default offset semantics.

        :return None: This test returns nothing.
        """
        model = SentenceEncoder(
            padding_idx=0,
            vocab_size=64,
            num_encoder_layers=1,
            embedding_dim=16,
            ffn_embedding_dim=32,
            num_attention_heads=2,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            max_seq_len=8,
            num_segments=0,
            encoder_normalize_before=True,
            activation_fn="gelu",
            normalize_before=False,
            relative_attention_num_buckets=8,
            relative_attention_max_distance=16,
        )
        model.eval()
        tokens = torch.tensor([[5, 6, 0, 0]])
        positions = torch.arange(tokens.size(1)).unsqueeze(0)
        with torch.no_grad():
            default_states, _ = model(tokens, last_state_only=True)
            explicit_states, _ = model(tokens, last_state_only=True, positions=positions)
        self.assertTrue(torch.allclose(default_states[0], explicit_states[0]))

    def test_sentence_encoder_positions_affect_relative_bias(self) -> None:
        """Ensure explicit positions influence relative position bias.

        :return None: This test returns nothing.
        """
        torch.manual_seed(0)
        model = SentenceEncoder(
            padding_idx=0,
            vocab_size=64,
            num_encoder_layers=1,
            embedding_dim=16,
            ffn_embedding_dim=32,
            num_attention_heads=2,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            max_seq_len=8,
            num_segments=0,
            use_position_embeddings=False,
            encoder_normalize_before=True,
            activation_fn="gelu",
            normalize_before=False,
            relative_attention_num_buckets=8,
            relative_attention_max_distance=16,
        )
        model.eval()
        tokens = torch.randint(1, 63, (1, 6))
        positions_a = torch.arange(tokens.size(1)).unsqueeze(0)
        positions_b = positions_a.flip(dims=[1])
        with torch.no_grad():
            states_a, _ = model(tokens, last_state_only=True, positions=positions_a)
            states_b, _ = model(tokens, last_state_only=True, positions=positions_b)
        self.assertFalse(torch.allclose(states_a[0], states_b[0]))

    def test_two_stream_attention_padding_mask(self) -> None:
        """Ensure padding masks stay 2D and SDPA matches non-SDPA behavior.

        :return None: This test returns nothing.
        """
        torch.manual_seed(0)
        bsz = 2
        base_seq_len = 4
        pred_size = 2
        input_len = base_seq_len + 2 * pred_size
        key_len = base_seq_len + pred_size

        input_ids = torch.randint(1, 10, (bsz, input_len))
        input_ids[0, 1] = 0
        input_ids[1, 2] = 0

        query_mask, content_mask, key_padding_mask = make_query_and_content_mask(
            input_ids, base_seq_len, pred_size, pad_token_id=0
        )
        self.assertEqual(query_mask.dim(), 2)
        self.assertEqual(content_mask.dim(), 2)
        self.assertIsNotNone(key_padding_mask)
        self.assertEqual(key_padding_mask.dim(), 2)
        self.assertEqual(key_padding_mask.shape, (bsz, key_len))
        self.assertTrue(key_padding_mask.any())

        embed_dim = 8
        num_heads = 2
        attn = RelativeMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            self_attention=True,
        )
        attn.eval()

        c = torch.randn(key_len, bsz, embed_dim)
        q = torch.randn(pred_size, bsz, embed_dim)

        attn.onnx_trace = False
        c_sdpa, q_sdpa = two_stream_self_attention(
            attn,
            query=[c, q],
            key=c,
            value=c,
            query_mask=query_mask,
            content_mask=content_mask,
            key_padding_mask=key_padding_mask,
        )

        attn.onnx_trace = True
        c_nosdpa, q_nosdpa = two_stream_self_attention(
            attn,
            query=[c, q],
            key=c,
            value=c,
            query_mask=query_mask,
            content_mask=content_mask,
            key_padding_mask=key_padding_mask,
        )

        torch.testing.assert_close(c_sdpa, c_nosdpa, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(q_sdpa, q_nosdpa, atol=1e-5, rtol=1e-4)

        attn.onnx_trace = False
        c_nomask, q_nomask = two_stream_self_attention(
            attn,
            query=[c, q],
            key=c,
            value=c,
            query_mask=query_mask,
            content_mask=content_mask,
            key_padding_mask=None,
        )
        self.assertFalse(torch.allclose(c_sdpa, c_nomask))
        self.assertFalse(torch.allclose(q_sdpa, q_nomask))

    def test_two_stream_mask_layout_matches_reference(self) -> None:
        """Ensure boolean mask layout matches the legacy float construction.

        :return None: This test returns nothing.
        """
        seq_len = 6
        pred_size = 2
        input_ids = torch.zeros(1, seq_len + pred_size, dtype=torch.long)
        query_mask, content_mask, key_padding_mask = make_query_and_content_mask(
            input_ids, seq_len, pred_size, pad_token_id=None, attention_mask=None
        )
        self.assertIsNone(key_padding_mask)
        self.assertEqual(query_mask.dtype, torch.bool)
        self.assertEqual(content_mask.dtype, torch.bool)

        device = input_ids.device
        mask = torch.triu(torch.ones(pred_size, pred_size, device=device), 0)
        ref_query = torch.cat(
            (
                torch.ones(pred_size, seq_len - pred_size, device=device),
                1 - mask,
                mask,
            ),
            dim=-1,
        ).eq(0)
        mask = torch.cat(
            [
                torch.zeros(seq_len - pred_size, pred_size, device=device),
                torch.tril(torch.ones(pred_size, pred_size, device=device), 0),
                torch.zeros(pred_size, pred_size, device=device),
            ],
            dim=0,
        )
        ref_content = torch.cat(
            (
                torch.ones(seq_len + pred_size, seq_len - pred_size, device=device),
                mask,
                1 - mask,
            ),
            dim=-1,
        ).eq(0)

        self.assertTrue(torch.equal(query_mask, ref_query))
        self.assertTrue(torch.equal(content_mask, ref_content))

    def test_resolve_best_loss_falls_back_to_best_checkpoint(self) -> None:
        """Fallback to best checkpoint best_loss when missing in resume checkpoint.

        :return None: This test returns nothing.
        """
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            best_checkpoint_path = checkpoint_dir / "best_checkpoint.pt"

            torch.save({"best_loss": 1.23}, best_checkpoint_path)

            best_loss = pretrain_mpnet._resolve_best_loss({"steps": 5}, checkpoint_dir)

            self.assertEqual(best_loss, 1.23)

    def test_resolve_best_loss_prefers_best_checkpoint_over_checkpoint(self) -> None:
        """Prefer best checkpoint best_loss over resume checkpoint value.

        :return None: This test returns nothing.
        """
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            best_checkpoint_path = checkpoint_dir / "best_checkpoint.pt"

            torch.save({"best_loss": 0.3}, best_checkpoint_path)

            best_loss = pretrain_mpnet._resolve_best_loss(
                {"steps": 5, "best_loss": 0.5}, checkpoint_dir
            )

            self.assertEqual(best_loss, 0.3)

    def test_get_resume_metadata_defaults_for_legacy(self) -> None:
        """Ensure legacy checkpoints default missing resume metadata.

        :return None: This test returns nothing.
        """
        checkpoint = {"steps": 1, "epoch": 0}
        samples_processed, data_state = pretrain_mpnet._get_resume_metadata(checkpoint, None)
        self.assertEqual(samples_processed, 0)
        self.assertTrue(data_state["legacy"])
        self.assertEqual(data_state["mode"], "legacy")
        self.assertEqual(data_state["cycle"], 0)
        self.assertEqual(data_state["batch_index"], 0)
        self.assertEqual(data_state["samples_in_cycle"], 0)

    def test_get_resume_metadata_prefers_data_state(self) -> None:
        """Ensure resume metadata uses explicit data_state when present.

        :return None: This test returns nothing.
        """
        checkpoint = {
            "samples_processed": 10,
            "data_state": {
                "mode": "streaming",
                "cycle": 2,
                "batch_index": 5,
                "samples_in_cycle": 128,
            },
        }
        samples_processed, data_state = pretrain_mpnet._get_resume_metadata(checkpoint, None)
        self.assertEqual(samples_processed, 10)
        self.assertEqual(data_state["mode"], "streaming")
        self.assertFalse(data_state["legacy"])
        self.assertEqual(data_state["cycle"], 2)
        self.assertEqual(data_state["batch_index"], 5)
        self.assertEqual(data_state["samples_in_cycle"], 128)

    def test_resolve_best_loss_prefers_resume_checkpoint_dir(self) -> None:
        """Prefer best checkpoint in resume checkpoint directory when external.

        :return None: This test returns nothing.
        """
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "new_run"
            resume_dir = Path(tmpdir) / "resume_run"
            checkpoint_dir.mkdir()
            resume_dir.mkdir()

            best_checkpoint_path = resume_dir / "best_checkpoint.pt"
            torch.save({"best_loss": 2.34}, best_checkpoint_path)

            resume_checkpoint = resume_dir / "checkpoint10.pt"
            best_loss = pretrain_mpnet._resolve_best_loss(
                {"steps": 10}, checkpoint_dir, resume_checkpoint
            )

        self.assertEqual(best_loss, 2.34)

    def test_select_resume_checkpoint_path_falls_back_to_latest(self) -> None:
        """Prefer latest interval checkpoint when no explicit resume checkpoint is provided.

        :return None: This test returns nothing.
        """
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            (checkpoint_dir / "checkpoint1.pt").write_bytes(b"")
            latest = checkpoint_dir / "checkpoint3.pt"
            latest.write_bytes(b"")

            selected = pretrain_mpnet._select_resume_checkpoint_path(checkpoint_dir, None)

            self.assertEqual(selected, latest)

    def test_select_resume_checkpoint_path_errors_on_missing(self) -> None:
        """Raise when no resume checkpoint exists.

        :return None: This test returns nothing.
        """
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            with self.assertRaises(FileNotFoundError):
                pretrain_mpnet._select_resume_checkpoint_path(checkpoint_dir, None)

    def test_model_summary_avoids_double_counting(self) -> None:
        """Ensure model_summary reports parent params without child double-counting.

        :return None: This test returns nothing.
        """

        class Parent(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.child = torch.nn.Linear(2, 3)

        model = Parent()
        expected_total = sum(p.numel() for p in model.parameters())

        from io import StringIO
        from contextlib import redirect_stdout

        buf = StringIO()
        with redirect_stdout(buf):
            utils.model_summary(model, max_depth=2)
        output = buf.getvalue().splitlines()
        parent_line = next(line for line in output if line.strip().startswith("Parent"))
        self.assertIn("--", parent_line)
        self.assertIn(f"Total params: {expected_total}", buf.getvalue())

    def test_pretraining_forward_return_mlm(self) -> None:
        """Ensure return_mlm path runs and yields logits with expected shapes.

        :return None: This test returns nothing.
        """
        args = Namespace(
            encoder_layers=2,
            encoder_embed_dim=32,
            encoder_ffn_dim=64,
            encoder_attention_heads=4,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.1,
            activation_fn="gelu",
            max_positions=16,
            relative_attention_num_buckets=8,
            relative_attention_max_distance=16,
            normalize_before=False,
            padded_vocab_size=30528,
        )
        tokenizer = DummyTokenizer()
        model = pretrain_mpnet.MPNetForPretraining(args, tokenizer)
        model.eval()

        batch_size = 2
        base_seq_len = 6
        pred_size = 2
        input_len = base_seq_len + 2 * pred_size
        input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, input_len))
        positions = torch.arange(input_len).unsqueeze(0).expand(batch_size, input_len)

        with torch.no_grad():
            logits, mlm_logits = model(input_ids, positions, pred_size, return_mlm=True)

        self.assertEqual(logits.shape[:2], (batch_size, pred_size))
        self.assertEqual(mlm_logits.shape[:2], (batch_size, pred_size))

    def test_pretraining_gradient_checkpointing_forward(self) -> None:
        """Ensure pretraining forward runs with gradient checkpointing enabled.

        :return None: This test returns nothing.
        """
        args = Namespace(
            encoder_layers=2,
            encoder_embed_dim=32,
            encoder_ffn_dim=64,
            encoder_attention_heads=4,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.1,
            activation_fn="gelu",
            max_positions=16,
            relative_attention_num_buckets=8,
            relative_attention_max_distance=16,
            normalize_before=False,
            padded_vocab_size=30528,
            gradient_checkpointing=True,
        )
        tokenizer = DummyTokenizer()
        model = pretrain_mpnet.MPNetForPretraining(args, tokenizer)
        model.train()

        batch_size = 2
        base_seq_len = 6
        pred_size = 2
        input_len = base_seq_len + 2 * pred_size
        input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, input_len))
        positions = torch.arange(input_len).unsqueeze(0).expand(batch_size, input_len)

        outs = model(input_ids, positions, pred_size, return_mlm=False)
        loss = outs.sum()
        loss.backward()
        self.assertIsNotNone(model.sentence_encoder.embed_tokens.weight.grad)

    def test_prune_checkpoints_keeps_recent(self) -> None:
        """Ensure checkpoint pruning keeps the most recent checkpoints.

        :return None: This test returns nothing.
        """
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            optimizer_dir = checkpoint_dir / "optimizer"
            optimizer_dir.mkdir()

            for step in range(1, 5):
                (checkpoint_dir / f"checkpoint{step}.pt").write_text("x")
                (optimizer_dir / f"checkpoint{step}_optimizer_state.pt").write_text("x")

            pretrain_mpnet._prune_checkpoints(checkpoint_dir, 2, optimizer_dir)

            remaining = sorted(p.name for p in checkpoint_dir.glob("checkpoint*.pt"))
            self.assertEqual(remaining, ["checkpoint3.pt", "checkpoint4.pt"])
            remaining_opt = sorted(
                p.name for p in optimizer_dir.glob("checkpoint*_optimizer_state.pt")
            )
            self.assertEqual(
                remaining_opt,
                ["checkpoint3_optimizer_state.pt", "checkpoint4_optimizer_state.pt"],
            )

    def test_select_architecture_source(self) -> None:
        """Verify architecture source selection precedence.

        :return None: This test returns nothing.
        """
        self.assertEqual(
            pretrain_mpnet._select_architecture_source(
                Namespace(resume=True, hf_model_path="hf/path")
            ),
            "hf",
        )
        self.assertEqual(
            pretrain_mpnet._select_architecture_source(
                Namespace(resume=False, hf_model_path="hf/path")
            ),
            "hf",
        )
        self.assertEqual(
            pretrain_mpnet._select_architecture_source(Namespace(resume=True, hf_model_path=None)),
            "resume",
        )
        self.assertEqual(
            pretrain_mpnet._select_architecture_source(Namespace(resume=False, hf_model_path=None)),
            "new",
        )

    def test_select_resume_checkpoint_path(self) -> None:
        """Check resume checkpoint path selection behavior.

        :return None: This test returns nothing.
        """
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            best_checkpoint = checkpoint_dir / "best_checkpoint.pt"
            best_checkpoint.write_bytes(b"")
            self.assertEqual(
                pretrain_mpnet._select_resume_checkpoint_path(checkpoint_dir, None),
                best_checkpoint,
            )
            latest_checkpoint = checkpoint_dir / "checkpoint42.pt"
            latest_checkpoint.write_bytes(b"")
            self.assertEqual(
                pretrain_mpnet._select_resume_checkpoint_path(checkpoint_dir, None),
                latest_checkpoint,
            )
            explicit_checkpoint = checkpoint_dir / "checkpoint123.pt"
            explicit_checkpoint.write_bytes(b"")
            self.assertEqual(
                pretrain_mpnet._select_resume_checkpoint_path(
                    checkpoint_dir, str(explicit_checkpoint)
                ),
                explicit_checkpoint,
            )

    def test_select_test_checkpoint_path_ignores_stale_best(self) -> None:
        """Ensure test eval does not reuse stale best checkpoints.

        :return None: This test returns nothing.
        """
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            stale_best = checkpoint_dir / "best_checkpoint.pt"
            stale_best.write_bytes(b"")

            self.assertIsNone(
                pretrain_mpnet._select_test_checkpoint_path(
                    checkpoint_dir,
                    best_checkpoint_written=False,
                )
            )

            self.assertEqual(
                pretrain_mpnet._select_test_checkpoint_path(
                    checkpoint_dir,
                    best_checkpoint_written=True,
                ),
                stale_best,
            )

    def test_select_optimizer_state_path(self) -> None:
        """Confirm optimizer state path matches resume checkpoint type.

        :return None: This test returns nothing.
        """
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            optimizer_dir = base_dir / "checkpoints" / "optimizer"
            best_checkpoint = base_dir / "checkpoints" / "best_checkpoint.pt"
            latest_checkpoint = base_dir / "checkpoints" / "checkpoint123.pt"

            self.assertEqual(
                pretrain_mpnet._select_optimizer_state_path(optimizer_dir, best_checkpoint),
                optimizer_dir / "best_optimizer_state.pt",
            )
            self.assertEqual(
                pretrain_mpnet._select_optimizer_state_path(optimizer_dir, latest_checkpoint),
                optimizer_dir / "checkpoint123_optimizer_state.pt",
            )

    def test_resolve_optimizer_state_dir(self) -> None:
        """Ensure optimizer state dir follows resume checkpoint location.

        :return None: This test returns nothing.
        """
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            checkpoint_dir = base_dir / "checkpoints"
            resume_checkpoint = checkpoint_dir / "best_checkpoint.pt"
            external_checkpoint = base_dir / "other_runs" / "best_checkpoint.pt"

            self.assertEqual(
                pretrain_mpnet._resolve_optimizer_state_dir(checkpoint_dir, resume_checkpoint),
                checkpoint_dir / "optimizer",
            )
            self.assertEqual(
                pretrain_mpnet._resolve_optimizer_state_dir(checkpoint_dir, external_checkpoint),
                external_checkpoint.parent / "optimizer",
            )

    def test_get_optimizer_state_path_for_resume(self) -> None:
        """Ensure optimizer state path resolves for internal/external checkpoints.

        :return None: This test returns nothing.
        """
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            checkpoint_dir = base_dir / "checkpoints"
            resume_checkpoint = checkpoint_dir / "best_checkpoint.pt"
            external_checkpoint = base_dir / "other_runs" / "checkpoint42.pt"

            self.assertEqual(
                pretrain_mpnet._get_optimizer_state_path_for_resume(
                    checkpoint_dir, resume_checkpoint
                ),
                checkpoint_dir / "optimizer" / "best_optimizer_state.pt",
            )
            self.assertEqual(
                pretrain_mpnet._get_optimizer_state_path_for_resume(
                    checkpoint_dir, external_checkpoint
                ),
                external_checkpoint.parent / "optimizer" / "checkpoint42_optimizer_state.pt",
            )

    def test_select_best_checkpoint_path_prefers_resume_root(self) -> None:
        """Fallback to resume root best checkpoint when local is missing.

        :return None: This test returns nothing.
        """
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "new_run"
            resume_dir = Path(tmpdir) / "resume_run"
            checkpoint_dir.mkdir()
            resume_dir.mkdir()

            resume_best = resume_dir / "best_checkpoint.pt"
            torch.save({"best_loss": 1.0}, resume_best)
            resume_checkpoint = resume_dir / "checkpoint10.pt"

            selected = pretrain_mpnet._select_best_checkpoint_path(
                checkpoint_dir, resume_checkpoint
            )
            self.assertEqual(selected, resume_best)

    def test_should_save_checkpoint(self) -> None:
        """Ensure checkpoint save decision is made at exact step boundaries.

        :return None: This test returns nothing.
        """
        self.assertFalse(pretrain_mpnet._should_save_checkpoint(0, 1000))
        self.assertFalse(pretrain_mpnet._should_save_checkpoint(999, 1000))
        self.assertTrue(pretrain_mpnet._should_save_checkpoint(1000, 1000))
        self.assertFalse(pretrain_mpnet._should_save_checkpoint(1000, 0))

    def test_strip_compile_prefix(self) -> None:
        """Ensure compile prefixes are removed from state dict keys.

        :return None: This test returns nothing.
        """
        state = {"_orig_mod.layer.weight": torch.tensor([1.0]), "layer.bias": torch.tensor([0.0])}
        stripped = pretrain_mpnet._strip_compile_prefix(state)
        self.assertIn("layer.weight", stripped)
        self.assertIn("layer.bias", stripped)
        self.assertNotIn("_orig_mod.layer.weight", stripped)

    def test_coerce_rng_state(self) -> None:
        """Ensure RNG state is coerced to uint8 CPU tensor.

        :return None: This test returns nothing.
        """
        state = torch.tensor([1, 2, 3], dtype=torch.int64)
        coerced = pretrain_mpnet._coerce_rng_state(state)
        self.assertEqual(coerced.dtype, torch.uint8)
        self.assertEqual(coerced.device.type, "cpu")
        self.assertTrue(torch.equal(coerced, torch.tensor([1, 2, 3], dtype=torch.uint8)))

    def test_apply_checkpoint_architecture_args_restores_max_tokens(self) -> None:
        """Ensure checkpoint args restore max_tokens and align max_positions.

        :return None: This test returns nothing.
        """
        args = Namespace(
            encoder_layers=1,
            encoder_embed_dim=64,
            encoder_ffn_dim=128,
            encoder_attention_heads=4,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.1,
            activation_fn="gelu",
            relative_attention_num_buckets=32,
            relative_attention_max_distance=64,
            normalize_before=False,
            original_vocab_size=100,
            padded_vocab_size=100,
            max_tokens=128,
            max_positions=128,
        )
        checkpoint_args = {
            "encoder_layers": 2,
            "encoder_embed_dim": 32,
            "encoder_ffn_dim": 64,
            "encoder_attention_heads": 2,
            "dropout": 0.2,
            "attention_dropout": 0.2,
            "activation_dropout": 0.2,
            "activation_fn": "relu",
            "relative_attention_num_buckets": 16,
            "relative_attention_max_distance": 128,
            "normalize_before": True,
            "original_vocab_size": 200,
            "padded_vocab_size": 256,
            "max_tokens": 256,
        }

        pretrain_mpnet._apply_checkpoint_architecture_args(args, checkpoint_args)

        self.assertEqual(args.max_tokens, 256)
        self.assertEqual(args.max_positions, 256)
        self.assertEqual(args.relative_attention_max_distance, 128)
        self.assertTrue(args.normalize_before)

    def test_apply_checkpoint_architecture_args_uses_legacy_rope_defaults(self) -> None:
        """Ensure missing RoPE/Flex fields fall back to legacy-safe defaults.

        :return None: This test returns nothing.
        """
        args = Namespace(
            encoder_layers=1,
            encoder_embed_dim=64,
            encoder_ffn_dim=128,
            encoder_attention_heads=4,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.1,
            activation_fn="gelu",
            relative_attention_num_buckets=32,
            relative_attention_max_distance=64,
            normalize_before=False,
            original_vocab_size=100,
            padded_vocab_size=128,
            max_tokens=128,
            max_positions=128,
            use_rope=True,
            rope_theta=500_000.0,
            rope_dim=32,
            rope_max_position_embeddings=8192,
            use_relative_attention_bias=False,
            use_flex_attention=False,
            flex_block_size=64,
            flex_compile_block_mask=True,
            gradient_checkpointing=False,
        )
        checkpoint_args = {
            "encoder_layers": 2,
            "encoder_embed_dim": 32,
            "encoder_ffn_dim": 64,
            "encoder_attention_heads": 2,
            "dropout": 0.2,
            "attention_dropout": 0.2,
            "activation_dropout": 0.2,
            "activation_fn": "relu",
            "relative_attention_num_buckets": 16,
            "relative_attention_max_distance": 128,
            "normalize_before": True,
            "original_vocab_size": 200,
            "padded_vocab_size": 256,
            "max_tokens": 256,
        }

        pretrain_mpnet._apply_checkpoint_architecture_args(args, checkpoint_args)

        self.assertFalse(args.use_rope)
        self.assertEqual(args.rope_theta, 10_000.0)
        self.assertIsNone(args.rope_dim)
        self.assertIsNone(args.rope_max_position_embeddings)
        self.assertTrue(args.use_relative_attention_bias)
        self.assertTrue(args.use_flex_attention)
        self.assertEqual(args.flex_block_size, 128)
        self.assertFalse(args.flex_compile_block_mask)

    def test_load_architecture_config_prefers_resume_root(self) -> None:
        """Ensure resume-root config is preferred over local checkpoint_dir config.

        :return None: This test returns nothing.
        """
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "current"
            checkpoint_dir.mkdir()
            resume_root = Path(tmpdir) / "resume_root"
            resume_root.mkdir()
            resume_checkpoint = resume_root / "checkpoint10.pt"
            resume_checkpoint.write_text("placeholder")

            with open(checkpoint_dir / "config.json", "w") as f:
                json.dump({"encoder_layers": 6}, f)
            with open(resume_root / "config.json", "w") as f:
                json.dump({"encoder_layers": 12}, f)

            config, config_path = pretrain_mpnet._load_architecture_config(
                checkpoint_dir, resume_checkpoint
            )

            self.assertEqual(config, {"encoder_layers": 12})
            self.assertEqual(config_path, resume_root / "config.json")

    def test_save_initial_run_outputs_writes_architecture_config(self) -> None:
        """Ensure initial run artifacts include config.json and training_args.json.

        :return None: This test returns nothing.
        """

        class TinyTokenizer:
            def save_pretrained(self, target_dir: Path) -> None:
                Path(target_dir).mkdir(parents=True, exist_ok=True)
                (Path(target_dir) / "tokenizer.json").write_text("{}")

        args = Namespace(
            encoder_layers=2,
            encoder_embed_dim=32,
            encoder_ffn_dim=64,
            encoder_attention_heads=2,
            dropout=0.1,
            attention_dropout=0.0,
            activation_dropout=0.1,
            activation_fn="gelu",
            relative_attention_num_buckets=16,
            relative_attention_max_distance=128,
            normalize_before=False,
            original_vocab_size=100,
            padded_vocab_size=128,
            max_tokens=128,
            max_positions=128,
            use_rope=True,
            rope_theta=10_000.0,
            rope_dim=None,
            rope_max_position_embeddings=None,
            use_relative_attention_bias=True,
            use_flex_attention=True,
            flex_block_size=128,
            flex_compile_block_mask=False,
            gradient_checkpointing=False,
            tokenizer_name="dummy",
            checkpoint_dir="./checkpoints",
            lr=1e-4,
        )

        with TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            pretrain_mpnet._save_initial_run_outputs(out_dir, args, TinyTokenizer())

            config_path = out_dir / "config.json"
            training_args_path = out_dir / "training_args.json"
            tokenizer_file = out_dir / "tokenizer" / "tokenizer.json"
            self.assertTrue(config_path.exists())
            self.assertTrue(training_args_path.exists())
            self.assertTrue(tokenizer_file.exists())

            with open(config_path) as f:
                config = json.load(f)
            self.assertTrue(config["use_rope"])
            self.assertEqual(config["encoder_layers"], 2)
            self.assertEqual(config["tokenizer_name"], "dummy")

    def test_normalize_training_accuracy(self) -> None:
        """Validate training accuracy normalization helper.

        :return None: This test returns nothing.
        """
        self.assertEqual(pretrain_mpnet._normalize_training_accuracy(0.0, 0), 0.0)
        self.assertAlmostEqual(pretrain_mpnet._normalize_training_accuracy(8.0, 10), 0.8)

    def test_accuracy_ignores_pad_tokens(self) -> None:
        """Ensure accuracy ignores padding tokens when requested.

        :return None: This test returns nothing.
        """
        logits = torch.tensor([[[0.1, 0.9], [2.0, 0.0]]])
        targets = torch.tensor([[1, 0]])
        self.assertEqual(pretrain_mpnet.accuracy(logits, targets), 2)
        self.assertEqual(
            pretrain_mpnet.accuracy(logits, targets, ignore_index=0),
            1,
        )

    def test_count_pred_tokens(self) -> None:
        """Check predicted token counting with padding.

        :return None: This test returns nothing.
        """
        targets = torch.tensor([[1, 0, 2], [0, 0, 3]])
        self.assertEqual(pretrain_mpnet._count_pred_tokens(targets, 0), 3)

    def test_autocast_context_cpu_is_noop(self) -> None:
        """Ensure CPU autocast context is a no-op.

        :return None: This test returns nothing.
        """
        with pretrain_mpnet._get_autocast_context(torch.device("cpu")):
            pass

    def test_ga_gradients_match_full_batch(self) -> None:
        """Verify GA gradients match full-batch gradients.

        :return None: This test returns nothing.
        """
        torch.manual_seed(0)
        vocab_size = 11
        seq_len = 5
        pad_token_id = 0

        logits = torch.randn(4, seq_len, vocab_size, requires_grad=True)
        targets = torch.randint(1, vocab_size, (4, seq_len))
        targets[0, -2:] = pad_token_id
        targets[2, -1:] = pad_token_id

        loss_full_sum = F.nll_loss(
            F.log_softmax(logits.view(-1, vocab_size), dim=-1),
            targets.view(-1),
            reduction="sum",
            ignore_index=pad_token_id,
        )
        total_tokens = pretrain_mpnet._count_pred_tokens(targets, pad_token_id)
        loss_full = loss_full_sum / total_tokens
        loss_full.backward()
        grads_full = logits.grad.clone()

        logits_ga = logits.detach().clone().requires_grad_(True)
        total_tokens_ga = 0
        for start in (0, 2):
            micro_logits = logits_ga[start : start + 2]
            micro_targets = targets[start : start + 2]
            loss_sum = F.nll_loss(
                F.log_softmax(micro_logits.view(-1, vocab_size), dim=-1),
                micro_targets.view(-1),
                reduction="sum",
                ignore_index=pad_token_id,
            )
            loss_sum.backward()
            total_tokens_ga += pretrain_mpnet._count_pred_tokens(micro_targets, pad_token_id)

        logits_ga.grad.div_(total_tokens_ga)
        self.assertTrue(torch.allclose(grads_full, logits_ga.grad, atol=1e-6, rtol=1e-5))

    def test_scheduler_state_dict_is_stateless(self) -> None:
        """Ensure scheduler state_dict is stateless and load accepts optimizer dicts.

        :return None: This test returns nothing.
        """
        args = Namespace(
            lr=1e-3,
            warmup_updates=1,
            end_learning_rate=0.0,
            total_updates=10,
            power=1.0,
        )
        param = torch.nn.Parameter(torch.randn(1))
        optimizer = torch.optim.AdamW([param], lr=args.lr)
        scheduler = PolynomialDecayLRScheduler(args, optimizer)

        self.assertTrue(scheduler.state_dict().get("stateless", False))
        scheduler.load_state_dict(optimizer.state_dict())


if __name__ == "__main__":
    unittest.main()
