"""Unit tests for pretraining helper utilities."""

import pathlib
import sys
import unittest
from argparse import Namespace
from tempfile import TemporaryDirectory

import torch
import torch.nn.functional as F

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
            "original_vocab_size": 200,
            "padded_vocab_size": 256,
            "max_tokens": 256,
        }

        pretrain_mpnet._apply_checkpoint_architecture_args(args, checkpoint_args)

        self.assertEqual(args.max_tokens, 256)
        self.assertEqual(args.max_positions, 256)

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
