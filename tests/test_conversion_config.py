"""Unit tests for conversion config.json helper behavior."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cli_tools.convert_hf_model_to_mpnet import _save_architecture_config
from cli_tools.convert_pretrained_mpnet_to_hf_model import (
    _build_mpnet_args_from_checkpoint_payload,
)


class TestConversionConfigHelpers(unittest.TestCase):
    """Tests for config sidecar read/write helpers used by conversion scripts."""

    def test_build_mpnet_args_prefers_config_json_over_checkpoint_args(self) -> None:
        """Prefer config.json architecture fields while preserving extra checkpoint metadata.

        :return None: This test returns nothing.
        """
        with TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint10.pt"
            checkpoint_path.write_text("placeholder")
            config_path = checkpoint_path.parent / "config.json"
            with open(config_path, "w") as f:
                json.dump({"encoder_layers": 6, "max_positions": 512}, f)

            state_dicts = {
                "args": {"encoder_layers": 12, "tokenizer_name": "legacy-tok", "pad_token_id": 0}
            }
            args = _build_mpnet_args_from_checkpoint_payload(checkpoint_path, state_dicts)

            self.assertEqual(args.encoder_layers, 6)
            self.assertEqual(args.max_positions, 512)
            self.assertEqual(args.tokenizer_name, "legacy-tok")
            self.assertEqual(args.pad_token_id, 0)

    def test_build_mpnet_args_falls_back_to_checkpoint_args(self) -> None:
        """Use checkpoint args when config.json is absent.

        :return None: This test returns nothing.
        """
        with TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint10.pt"
            checkpoint_path.write_text("placeholder")
            state_dicts = {"args": {"encoder_layers": 4, "max_positions": 256}}
            args = _build_mpnet_args_from_checkpoint_payload(checkpoint_path, state_dicts)

            self.assertEqual(args.encoder_layers, 4)
            self.assertEqual(args.max_positions, 256)

    def test_build_mpnet_args_raises_when_no_sources_exist(self) -> None:
        """Raise when neither config.json nor checkpoint args are available.

        :return None: This test returns nothing.
        """
        with TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint10.pt"
            checkpoint_path.write_text("placeholder")
            with self.assertRaises(KeyError):
                _build_mpnet_args_from_checkpoint_payload(checkpoint_path, {})

    def test_save_architecture_config_writes_json_payload(self) -> None:
        """Write config.json alongside converted checkpoint artifacts.

        :return None: This test returns nothing.
        """
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            args = Namespace(encoder_layers=8, max_positions=512, max_tokens=512)
            config_path = _save_architecture_config(checkpoint_dir, args)

            self.assertEqual(config_path, checkpoint_dir / "config.json")
            self.assertTrue(config_path.exists())
            with open(config_path) as f:
                payload = json.load(f)
            self.assertEqual(payload["encoder_layers"], 8)
            self.assertEqual(payload["max_positions"], 512)
            self.assertEqual(payload["max_tokens"], 512)


if __name__ == "__main__":
    unittest.main()
