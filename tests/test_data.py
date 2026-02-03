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
        self.assertEqual(permuted_examples["pred_size"], 1)
        self.assertEqual(permuted_examples["input_ids"].shape[0], 2)
        self.assertEqual(permuted_examples["targets"].shape[0], 2)
        self.assertIn("pred_ntokens", permuted_examples)
        self.assertEqual(
            permuted_examples["pred_ntokens"],
            int(permuted_examples["targets"].ne(self.collator.tokenizer.pad_token_id).sum().item()),
        )

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

    def test_collator_skips_special_and_padding_targets(self) -> None:
        """Ensure targets exclude special/pad tokens and attention_mask matches padding.

        :return None: This test returns nothing.
        """
        tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")
        enc_a = tokenizer("hello world", add_special_tokens=True)
        enc_b = tokenizer("short", add_special_tokens=True)
        examples = [
            {"input_ids": torch.tensor(enc_a["input_ids"], dtype=torch.long)},
            {"input_ids": torch.tensor(enc_b["input_ids"], dtype=torch.long)},
        ]
        collator = DataCollatorForMaskedPermutedLanguageModeling(
            tokenizer=tokenizer, random_seed=123
        )
        batch = collator.collate_fn(examples)

        targets = batch["targets"]
        special_ids = set(tokenizer.all_special_ids)
        special_ids.add(tokenizer.pad_token_id)
        for sid in special_ids:
            self.assertFalse(torch.any(targets.eq(sid)))

        attention_mask = batch["attention_mask"]
        input_ids = batch["input_ids"]
        self.assertEqual(attention_mask.shape, input_ids.shape)
        pad_mask = input_ids.eq(tokenizer.pad_token_id)
        self.assertTrue(torch.equal(attention_mask.eq(0), pad_mask))

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

    def test_streaming_dataset_defers_padding_to_collator(self) -> None:
        """Ensure streaming dataset does not pad to max_length.

        :return None: This test returns nothing.
        """
        tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")
        dataset_stream = [{"text": "hi"}]
        dataset = HFStreamingDataset(
            tokenizer=tokenizer,
            dataset_stream=dataset_stream,
            block_size=16,
            buffer_size=1,
            seed=0,
            min_text_length=0,
            skip_samples=0,
            max_samples=1,
            text_field="text",
        )
        sample = next(iter(dataset))
        self.assertLess(len(sample["input_ids"]), 16)

    def test_collator_rng_state_roundtrip(self) -> None:
        """Ensure collator RNG state restores deterministic masking.

        :return None: This test returns nothing.
        """
        tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")
        collator = DataCollatorForMaskedPermutedLanguageModeling(
            tokenizer=tokenizer, random_seed=12345
        )
        _ = collator.collate_fn(self.examples)
        state = collator.get_rng_state()
        self.assertIsNotNone(state)

        batch_a = collator.collate_fn(self.examples)
        collator.set_rng_state(state)
        batch_b = collator.collate_fn(self.examples)
        self.assertTrue(torch.equal(batch_a["input_ids"], batch_b["input_ids"]))
        self.assertTrue(torch.equal(batch_a["positions"], batch_b["positions"]))

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

    def test_streaming_dataset_rng_state_roundtrip(self) -> None:
        """Ensure streaming dataset RNG state is a no-op.

        :return None: This test returns nothing.
        """
        tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")
        mock_stream = [
            {"text": f"This is sample number {i} with enough text for testing."} for i in range(50)
        ]

        dataset = HFStreamingDataset(
            tokenizer=tokenizer,
            dataset_stream=mock_stream,
            block_size=32,
            buffer_size=10,
            seed=12345,
            min_text_length=0,
        )

        state = dataset.get_rng_state()
        self.assertIsNone(state)

        # Verify set_rng_state remains a no-op
        dataset.set_rng_state(state)
