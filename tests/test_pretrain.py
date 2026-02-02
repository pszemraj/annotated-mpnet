"""Unit tests for pretraining helper utilities."""

import pathlib
import sys
import unittest
from argparse import Namespace
from tempfile import TemporaryDirectory

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from annotated_mpnet.scheduler import PolynomialDecayLRScheduler
from annotated_mpnet.transformer_modules import SentenceEncoder
from annotated_mpnet.utils import utils

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cli_tools import pretrain_mpnet


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
        tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")
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

    def test_resolve_best_loss_falls_back_to_best_checkpoint(self) -> None:
        """Fallback to best checkpoint best_loss when missing in resume checkpoint.

        :return None: This test returns nothing.
        """
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = pathlib.Path(tmpdir)
            best_checkpoint_path = checkpoint_dir / "best_checkpoint.pt"

            torch.save({"best_loss": 1.23}, best_checkpoint_path)

            best_loss = pretrain_mpnet._resolve_best_loss({"steps": 5}, checkpoint_dir)

            self.assertEqual(best_loss, 1.23)

    def test_resolve_best_loss_prefers_best_checkpoint_over_checkpoint(self) -> None:
        """Prefer best checkpoint best_loss over resume checkpoint value.

        :return None: This test returns nothing.
        """
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = pathlib.Path(tmpdir)
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
            checkpoint_dir = pathlib.Path(tmpdir) / "new_run"
            resume_dir = pathlib.Path(tmpdir) / "resume_run"
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
        """Fallback to latest interval checkpoint when best is missing.

        :return None: This test returns nothing.
        """
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = pathlib.Path(tmpdir)
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
            checkpoint_dir = pathlib.Path(tmpdir)

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
        tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")
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
        tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")
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
            checkpoint_dir = pathlib.Path(tmpdir)
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
            checkpoint_dir = pathlib.Path(tmpdir)
            best_checkpoint = checkpoint_dir / "best_checkpoint.pt"
            best_checkpoint.write_bytes(b"")
            self.assertEqual(
                pretrain_mpnet._select_resume_checkpoint_path(checkpoint_dir, None),
                best_checkpoint,
            )
            explicit_checkpoint = checkpoint_dir / "checkpoint123.pt"
            explicit_checkpoint.write_bytes(b"")
            self.assertEqual(
                pretrain_mpnet._select_resume_checkpoint_path(
                    checkpoint_dir, str(explicit_checkpoint)
                ),
                explicit_checkpoint,
            )

    def test_select_optimizer_state_path(self) -> None:
        """Confirm optimizer state path matches resume checkpoint type.

        :return None: This test returns nothing.
        """
        optimizer_dir = pathlib.Path("/tmp/checkpoints/optimizer")
        best_checkpoint = pathlib.Path("/tmp/checkpoints/best_checkpoint.pt")
        latest_checkpoint = pathlib.Path("/tmp/checkpoints/checkpoint123.pt")

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
        checkpoint_dir = pathlib.Path("/tmp/checkpoints")
        resume_checkpoint = checkpoint_dir / "best_checkpoint.pt"
        external_checkpoint = pathlib.Path("/tmp/other_runs/best_checkpoint.pt")

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
        checkpoint_dir = pathlib.Path("/tmp/checkpoints")
        resume_checkpoint = checkpoint_dir / "best_checkpoint.pt"
        external_checkpoint = pathlib.Path("/tmp/other_runs/checkpoint42.pt")

        self.assertEqual(
            pretrain_mpnet._get_optimizer_state_path_for_resume(checkpoint_dir, resume_checkpoint),
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
            checkpoint_dir = pathlib.Path(tmpdir) / "new_run"
            resume_dir = pathlib.Path(tmpdir) / "resume_run"
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
