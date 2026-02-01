import pathlib
import sys
import unittest
from contextlib import contextmanager
from tempfile import TemporaryDirectory

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cli_tools import pretrain_mpnet


@contextmanager
def _argv_ctx(args):
    original = sys.argv[:]
    sys.argv = args
    try:
        yield
    finally:
        sys.argv = original


class TestPretrainSmoke(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_resume_smoke_run(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            data_dir = tmp_path / "data"
            train_dir = data_dir / "train"
            train_dir.mkdir(parents=True, exist_ok=True)
            valid_file = data_dir / "valid.txt"
            test_file = data_dir / "test.txt"
            checkpoint_dir = tmp_path / "checkpoints"

            (train_dir / "train.txt").write_text(
                "The quick brown fox jumps over the lazy dog.\n"
                "Pack my box with five dozen liquor jugs.\n"
            )
            valid_file.write_text(
                "Sphinx of black quartz, judge my vow.\nHow vexingly quick daft zebras jump.\n"
            )
            test_file.write_text(
                "Bright vixens jump; dozy fowl quack.\nQuick zephyrs blow, vexing daft Jim.\n"
            )

            base_args = [
                "pretrain_mpnet",
                "--dataset-name",
                "",
                "--train-dir",
                f"{train_dir}/",
                "--valid-file",
                str(valid_file),
                "--test-file",
                str(test_file),
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--save-optimizer-state",
                "--checkpoint-interval",
                "1",
                "--total-updates",
                "1",
                "--update-freq",
                "1",
                "--batch-size",
                "2",
                "--max-tokens",
                "32",
                "--max-positions",
                "32",
                "--encoder-layers",
                "2",
                "--encoder-embed-dim",
                "64",
                "--encoder-ffn-dim",
                "128",
                "--encoder-attention-heads",
                "4",
                "--min-text-length",
                "1",
                "--num-workers",
                "0",
            ]

            with _argv_ctx(base_args):
                pretrain_mpnet.cli_main()

            best_checkpoint = checkpoint_dir / "best_checkpoint.pt"
            best_optimizer = checkpoint_dir / "optimizer" / "best_optimizer_state.pt"
            self.assertTrue(best_checkpoint.exists())
            self.assertTrue(best_optimizer.exists())

            resume_args = base_args + [
                "--resume",
                "--resume-checkpoint",
                str(best_checkpoint),
            ]

            with _argv_ctx(resume_args):
                pretrain_mpnet.cli_main()


if __name__ == "__main__":
    unittest.main()
