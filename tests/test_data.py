"""
Tests for the collator function in the mpnet_data module
"""

import unittest

import torch
from transformers import AutoTokenizer

from annotated_mpnet.data import mpnet_data
from annotated_mpnet.data.mpnet_data import (
    DataCollatorForMaskedPermutedLanguageModeling,
    HFStreamingDataset,
    RandomSamplerWithSeed,
)


class TestData(unittest.TestCase):
    """Unit tests for MPNet data utilities."""

    def setUp(self) -> None:
        """Set up common fixtures for data tests.

        :return None: This method returns nothing.
        """
        self.examples = [
            {
                "input_ids": torch.tensor(
                    [0, 2027, 2007, 2023, 2746, 2001, 1041, 2311, 3235, 1003, 2],
                ),
            },
            {
                "input_ids": torch.tensor(
                    [0, 2027, 2007, 2182, 3235, 2],
                )
            },
        ]

        tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")

        self.collator = DataCollatorForMaskedPermutedLanguageModeling(
            tokenizer=tokenizer, random_seed=12345
        )

    def test_permuted_batch(self) -> None:
        """Validate permuted batch outputs from the collator.

        :return None: This test returns nothing.
        """
        permuted_examples = self.collator.collate_fn(self.examples)
        self.assertEqual(permuted_examples["pred_size"], 2)
        self.assertEqual(permuted_examples["input_ids"].shape[0], 2)
        self.assertEqual(permuted_examples["targets"].shape[0], 2)

        # Same seed should produce deterministic first batches.
        collator_b = DataCollatorForMaskedPermutedLanguageModeling(
            tokenizer=self.collator.tokenizer, random_seed=12345
        )
        permuted_examples_b = collator_b.collate_fn(self.examples)
        self.assertTrue(
            torch.equal(permuted_examples["input_ids"], permuted_examples_b["input_ids"]),
            "Seeded collator produced different first batch inputs",
        )
        self.assertTrue(
            torch.equal(permuted_examples["positions"], permuted_examples_b["positions"]),
            "Seeded collator produced different first batch positions",
        )

        # Consecutive calls should not repeat identical masking/permutations.
        permuted_examples_next = self.collator.collate_fn(self.examples)
        self.assertFalse(
            torch.equal(permuted_examples["positions"], permuted_examples_next["positions"]),
            "Masking/permutation repeated identically across batches with a fixed seed",
        )

    def test_training_seeded_sampling(self) -> None:
        """Verify deterministic sampling across epochs with a seed.

        :return None: This test returns nothing.
        """
        sampler_epoch_0 = RandomSamplerWithSeed([self.examples] * 10, epoch=0, random_seed=12345)
        sampler_epoch_1 = RandomSamplerWithSeed([self.examples] * 10, epoch=1, random_seed=12345)

        self.assertEqual(
            list(sampler_epoch_0.__iter__()),
            [0, 7, 3, 9, 6, 4, 1, 8, 5, 2],
            "Sampler not seeding correctly",
        )
        self.assertNotEqual(
            list(sampler_epoch_0.__iter__()),
            list(sampler_epoch_1.__iter__()),
            "Sampler seed may be broken since epoch 0 and epoch 1 are showing the same smpl order",
        )

    def test_streaming_dataset_skip_and_max(self) -> None:
        """Validate skip/max sample handling for streaming datasets.

        :return None: This test returns nothing.
        """
        tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")
        dataset_stream = [{"text": f"sample {i}"} for i in range(10)]
        dataset = HFStreamingDataset(
            tokenizer=tokenizer,
            dataset_stream=dataset_stream,
            block_size=8,
            buffer_size=2,
            seed=0,
            min_text_length=0,
            skip_samples=2,
            max_samples=3,
            text_field="text",
        )
        samples = list(iter(dataset))
        self.assertEqual(len(samples), 3)

    def test_collator_falls_back_without_fast_perm(self) -> None:
        """Ensure collator disables fast path when extension is missing.

        :return None: This test returns nothing.
        """
        tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")
        original = mpnet_data.make_span_perm
        try:
            mpnet_data.make_span_perm = None
            collator = DataCollatorForMaskedPermutedLanguageModeling(
                tokenizer=tokenizer, use_fast=True
            )
            self.assertFalse(collator.use_fast)
        finally:
            mpnet_data.make_span_perm = original
