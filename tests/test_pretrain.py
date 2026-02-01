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
            pretrain_mpnet._select_architecture_source(Namespace(resume=True, hf_model_path=None)),
            "resume",
        )
        self.assertEqual(
            pretrain_mpnet._select_architecture_source(Namespace(resume=False, hf_model_path=None)),
            "new",
        )


if __name__ == "__main__":
    unittest.main()
