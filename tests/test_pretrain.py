import pathlib
import sys
import unittest
from argparse import Namespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cli_tools import pretrain_mpnet


class TestPretrainHelpers(unittest.TestCase):
    def test_get_initial_best_loss(self) -> None:
        self.assertEqual(
            pretrain_mpnet._get_initial_best_loss(None), pretrain_mpnet.DEFAULT_BEST_LOSS
        )
        self.assertEqual(pretrain_mpnet._get_initial_best_loss({"best_loss": 1.23}), 1.23)
        self.assertEqual(
            pretrain_mpnet._get_initial_best_loss({"steps": 5}),
            pretrain_mpnet.DEFAULT_BEST_LOSS,
        )

    def test_select_architecture_source(self) -> None:
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
        optimizer_dir = pathlib.Path("/tmp/checkpoints/optimizer")
        best_checkpoint = pathlib.Path("/tmp/checkpoints/best_checkpoint.pt")
        latest_checkpoint = pathlib.Path("/tmp/checkpoints/checkpoint123.pt")

        self.assertEqual(
            pretrain_mpnet._select_optimizer_state_path(optimizer_dir, best_checkpoint),
            optimizer_dir / "best_optimizer_state.pt",
        )
        self.assertEqual(
            pretrain_mpnet._select_optimizer_state_path(optimizer_dir, latest_checkpoint),
            optimizer_dir / "optimizer_state.pt",
        )

    def test_normalize_training_accuracy(self) -> None:
        self.assertEqual(pretrain_mpnet._normalize_training_accuracy(0.0, 0), 0.0)
        self.assertAlmostEqual(pretrain_mpnet._normalize_training_accuracy(8.0, 10), 0.8)


if __name__ == "__main__":
    unittest.main()
