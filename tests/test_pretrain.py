"""Unit tests for pretraining helper utilities."""

import pathlib
import sys
import unittest
from argparse import Namespace
from tempfile import TemporaryDirectory

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

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
        checkpoint_dir = pathlib.Path("/tmp/checkpoints")
        self.assertEqual(
            pretrain_mpnet._select_resume_checkpoint_path(checkpoint_dir, None),
            checkpoint_dir / "best_checkpoint.pt",
        )
        self.assertEqual(
            pretrain_mpnet._select_resume_checkpoint_path(
                checkpoint_dir, "/tmp/checkpoints/checkpoint123.pt"
            ),
            pathlib.Path("/tmp/checkpoints/checkpoint123.pt"),
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


if __name__ == "__main__":
    unittest.main()
