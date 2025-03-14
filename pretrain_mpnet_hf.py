#!/usr/bin/env python3

"""
Script for pretraining MPNet using HuggingFace's Trainer API
"""

import argparse
import logging
import os
import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.modeling_outputs import MaskedLMOutput

from annotated_mpnet.data import (
    DataCollatorForMaskedPermutedLanguageModeling,
    MPNetDataset,
)
from annotated_mpnet.modeling import MPNetForPretraining

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to the model architecture.
    """

    encoder_layers: int = field(
        default=12, metadata={"help": "Number of encoder layers in the model"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "Dimension of the embedding layer"}
    )
    encoder_ffn_dim: int = field(
        default=3072, metadata={"help": "Dimension of the feed-forward hidden layer"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "Number of attention heads per layer"}
    )
    dropout: float = field(default=0.1, metadata={"help": "Dropout probability"})
    attention_dropout: float = field(
        default=0.1, metadata={"help": "Attention dropout probability"}
    )
    activation_dropout: float = field(
        default=0.1, metadata={"help": "Activation dropout probability"}
    )
    max_positions: int = field(
        default=512, metadata={"help": "Maximum number of positions in the model"}
    )
    activation_fn: str = field(
        default="gelu", metadata={"help": "Activation function used in the model"}
    )
    normalize_before: bool = field(
        default=False,
        metadata={"help": "Apply layer norm before attention calculation"},
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to the data processing.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the HuggingFace dataset to use"}
    )
    text_field: str = field(
        default="text", metadata={"help": "The field containing text in the dataset"}
    )
    train_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory containing training files"}
    )
    valid_file: Optional[str] = field(
        default=None, metadata={"help": "File containing validation data"}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "File containing test data"}
    )
    buffer_size: int = field(
        default=10000, metadata={"help": "Size of buffer for streaming datasets"}
    )
    eval_samples: int = field(
        default=500, metadata={"help": "Number of samples for evaluation"}
    )
    min_text_length: int = field(
        default=64, metadata={"help": "Minimum text length to consider"}
    )
    max_seq_length: int = field(
        default=512, metadata={"help": "Maximum sequence length for the model"}
    )
    pred_prob: float = field(
        default=0.15,
        metadata={"help": "Probability that a token will be in the prediction section"},
    )
    keep_prob: float = field(
        default=0.10,
        metadata={"help": "Probability that a token in the pred will be kept as is"},
    )
    rand_prob: float = field(
        default=0.10,
        metadata={
            "help": "Probability that a token in the pred will be randomly corrupted"
        },
    )
    whole_word_mask: bool = field(
        default=True,
        metadata={
            "help": "Whether to do whole word masking when generating permutations"
        },
    )


class MPNetForPretrainingHF(torch.nn.Module):
    """
    Adapter class that wraps MPNetForPretraining to make it compatible with HuggingFace's Trainer.
    """

    def __init__(self, args, tokenizer):
        super().__init__()
        self.model = MPNetForPretraining(args, tokenizer)
        self.tokenizer = tokenizer

    def forward(self, input_ids, positions, labels=None, **kwargs):
        # Forward pass through the model
        # We don't need pred_size anymore as it's not included in the batch
        # Using a fixed value based on the labels tensor size
        pred_size = labels.size(1) if labels is not None else None

        logits = self.model(
            input_ids=input_ids,
            positions=positions,
            pred_size=pred_size,
            return_mlm=False,
        )

        loss = None
        if labels is not None:
            # Calculate loss
            loss = F.nll_loss(
                F.log_softmax(
                    logits.view(-1, logits.size(-1)), dim=-1, dtype=torch.float32
                ),
                labels.view(-1),
                reduction="mean",
                ignore_index=self.tokenizer.pad_token_id,
            )

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
        )


class MPNetDataCollatorForHFTrainer(DataCollatorForMaskedPermutedLanguageModeling):
    """
    Adapter for DataCollatorForMaskedPermutedLanguageModeling to make it compatible with HF Trainer.
    Ensures all batch elements are proper tensors for Trainer compatibility.
    """

    def __call__(self, examples):
        batch = super().__call__(examples)

        # Rename targets to labels for the Trainer
        batch["labels"] = batch.pop("targets")

        # Convert scalar values to tensors to avoid concatenation errors
        # The Trainer expects all batch elements to be tensors
        new_batch = {}
        for k, v in batch.items():
            if k in ["pred_size", "ntokens"]:
                # Skip these fields as they're not needed by the Trainer
                continue
            elif not isinstance(v, torch.Tensor):
                # Convert any non-tensor to a tensor
                if isinstance(v, (int, float)):
                    new_batch[k] = torch.tensor(v, dtype=torch.long)
                else:
                    new_batch[k] = torch.tensor(v)
            else:
                new_batch[k] = v

        return new_batch


class MPNetStreamingDataset(IterableDataset):
    """
    An iterable dataset that uses HF's streaming functionality with MPNet's permutation approach.
    """

    def __init__(
        self,
        tokenizer,
        dataset_name,
        split="train",
        block_size=512,
        buffer_size=10000,
        seed=42,
        min_text_length=200,
        text_field="text",
    ):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.split = split
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.seed = seed
        self.min_text_length = min_text_length
        self.text_field = text_field

        # Initialize the streaming dataset
        from datasets import load_dataset

        self.dataset = load_dataset(dataset_name, split=split, streaming=True)

        # Apply filter for minimum text length
        if min_text_length > 0:
            self.dataset = self.dataset.filter(
                lambda ex: len(ex[text_field]) >= min_text_length
            )

        # Set random seed for reproducible shuffling
        random.seed(seed)

    def __iter__(self):
        # Set worker seed for proper sharding
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            # Use different seed for each worker
            worker_seed = worker_info.id + self.seed
            random.seed(worker_seed)

            # Shard the dataset based on worker id
            dataset_iter = iter(
                self.dataset.shuffle(
                    buffer_size=self.buffer_size, seed=worker_seed
                ).shard(num_shards=worker_info.num_workers, index=worker_info.id)
            )
        else:
            # No worker sharding needed
            dataset_iter = iter(
                self.dataset.shuffle(buffer_size=self.buffer_size, seed=self.seed)
            )

        # Initialize buffer
        buffer = []

        # Fill the buffer initially
        for _ in range(
            min(self.buffer_size, 1000)
        ):  # Use a smaller initial buffer to start faster
            try:
                example = next(dataset_iter)
                buffer.append(example)
            except StopIteration:
                break

        # Continue as long as there are items in buffer
        while buffer:
            # Randomly select an example from the buffer
            idx = random.randint(0, len(buffer) - 1)
            example = buffer[idx]

            # Process the example
            processed = self._process_example(example)

            # Replace the used example with a new one if available
            try:
                buffer[idx] = next(dataset_iter)
            except StopIteration:
                # Remove the used example if no more examples
                buffer.pop(idx)

            # Only yield non-empty examples
            if processed and len(processed["input_ids"]) > 0:
                yield processed

    def _process_example(self, example):
        """Process a single example from the dataset"""
        text = example[self.text_field]

        # Skip empty texts
        if not text or len(text) < self.min_text_length:
            return None

        try:
            # Tokenize the text - ensuring we get valid tensors
            encoding = self.tokenizer(
                text,
                max_length=self.block_size,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            # Return the input_ids in the expected format
            return {"input_ids": encoding["input_ids"].squeeze(0)}
        except Exception as e:
            # Log the error but don't crash - this helps with robustness
            print(f"Error processing text: {e}")
            return None


class MPNetTrainer(Trainer):
    """
    Custom Trainer class for MPNet that handles the specific training behavior.
    """

    def compute_loss(
        self, model, inputs, num_items_in_batch=None, return_outputs=False
    ):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # Forward pass
        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss / num_items_in_batch, outputs) if return_outputs else loss


def get_datasets(data_args, tokenizer):
    """Load and prepare datasets for training and evaluation"""

    if data_args.dataset_name is not None:
        # Use HuggingFace datasets
        from datasets import load_dataset

        try:
            # Load the streaming dataset for validation and test
            full_stream = load_dataset(
                data_args.dataset_name, split="train", streaming=True
            )

            # Apply minimum text length filter
            if data_args.min_text_length > 0:
                full_stream = full_stream.filter(
                    lambda ex: len(ex[data_args.text_field])
                    >= data_args.min_text_length
                )

            # Take samples for validation and test
            valid_examples = []
            valid_iter = iter(full_stream.take(data_args.eval_samples))
            for _ in range(data_args.eval_samples):
                try:
                    valid_examples.append(next(valid_iter))
                except StopIteration:
                    logger.warning(
                        f"Could only get {len(valid_examples)} examples for validation"
                    )
                    break

            test_examples = []
            test_iter = iter(
                full_stream.skip(data_args.eval_samples).take(data_args.eval_samples)
            )
            for _ in range(data_args.eval_samples):
                try:
                    test_examples.append(next(test_iter))
                except StopIteration:
                    logger.warning(
                        f"Could only get {len(test_examples)} examples for testing"
                    )
                    break

            # Create validation and test datasets
            valid_dataset = MPNetDataset(
                tokenizer=tokenizer,
                dataset=valid_examples,
                block_size=data_args.max_seq_length,
                field_name=data_args.text_field,
            )

            test_dataset = MPNetDataset(
                tokenizer=tokenizer,
                dataset=test_examples,
                block_size=data_args.max_seq_length,
                field_name=data_args.text_field,
            )

            # Create the streaming training dataset
            train_dataset = MPNetStreamingDataset(
                tokenizer=tokenizer,
                dataset_name=data_args.dataset_name,
                split="train",
                block_size=data_args.max_seq_length,
                buffer_size=data_args.buffer_size,
                seed=42,
                min_text_length=data_args.min_text_length,
                text_field=data_args.text_field,
            )

            return train_dataset, valid_dataset, test_dataset

        except Exception as e:
            logger.error(f"Error loading dataset {data_args.dataset_name}: {e}")
            raise
    else:
        # Use file-based datasets
        if not (data_args.train_dir and data_args.valid_file and data_args.test_file):
            raise ValueError(
                "Either dataset_name or (train_dir, valid_file, test_file) must be provided"
            )

        # Load validation and test datasets
        valid_dataset = MPNetDataset(
            tokenizer=tokenizer,
            file_path=data_args.valid_file,
            block_size=data_args.max_seq_length,
        )

        test_dataset = MPNetDataset(
            tokenizer=tokenizer,
            file_path=data_args.test_file,
            block_size=data_args.max_seq_length,
        )

        # Get list of training files
        train_files = [
            os.path.join(data_args.train_dir, f)
            for f in os.listdir(data_args.train_dir)
            if os.path.isfile(os.path.join(data_args.train_dir, f))
        ]

        if not train_files:
            raise ValueError(f"No training files found in {data_args.train_dir}")

        # Use the first file for the training dataset
        train_dataset = MPNetDataset(
            tokenizer=tokenizer,
            file_path=train_files[0],
            block_size=data_args.max_seq_length,
        )

        return train_dataset, valid_dataset, test_dataset


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Check for last checkpoint
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")

    # Check and adjust vocab_size parameter for better GPU performance
    original_vocab_size = tokenizer.vocab_size
    target_vocab_size = (
        (original_vocab_size + 127) // 128
    ) * 128  # Round up to nearest multiple of 128

    if target_vocab_size > original_vocab_size:
        logger.info(
            f"Padding model's vocab_size from {original_vocab_size} to {target_vocab_size} "
            "(div. by 128) for GPU performance"
        )
        padded_vocab_size = target_vocab_size
    else:
        padded_vocab_size = original_vocab_size

    # Create data collator for masked permuted language modeling
    data_collator = MPNetDataCollatorForHFTrainer(
        tokenizer=tokenizer,
        pred_prob=data_args.pred_prob,
        keep_prob=data_args.keep_prob,
        rand_prob=data_args.rand_prob,
        whole_word_mask=data_args.whole_word_mask,
    )

    # Load datasets
    train_dataset, eval_dataset, test_dataset = get_datasets(data_args, tokenizer)

    # Initialize model arguments as a Namespace
    model_namespace = argparse.Namespace(
        encoder_layers=model_args.encoder_layers,
        encoder_embed_dim=model_args.encoder_embed_dim,
        encoder_ffn_dim=model_args.encoder_ffn_dim,
        encoder_attention_heads=model_args.encoder_attention_heads,
        dropout=model_args.dropout,
        attention_dropout=model_args.attention_dropout,
        activation_dropout=model_args.activation_dropout,
        max_positions=model_args.max_positions,
        activation_fn=model_args.activation_fn,
        normalize_before=model_args.normalize_before,
        padded_vocab_size=padded_vocab_size,
    )

    # Create model
    model = MPNetForPretrainingHF(model_namespace, tokenizer)

    # Define compute_metrics function for the trainer
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = torch.argmax(torch.tensor(logits), dim=-1)

        # Convert to tensors if they're not already
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.tensor(predictions)

        # Only consider non-padding tokens
        mask = labels != tokenizer.pad_token_id
        labels = labels[mask]
        predictions = predictions[mask]

        correct = (predictions == labels).sum().item()
        total = labels.size(0)

        return {"accuracy": correct / total if total > 0 else 0}

    training_args.save_safetensors = False  # Save model in safetensors format
    training_args.ddp_find_unused_parameters = (
        False  # Disable unused parameters warning
    )
    training_args.include_num_input_tokens_seen = True
    training_args.gradient_checkpointing = True

    # Initialize trainer
    trainer = MPNetTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Testing
    if training_args.do_predict:
        logger.info("*** Test ***")
        metrics = trainer.evaluate(test_dataset)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


def check_tf32_capability():
    """Check if the GPU supports NVIDIA Ampere or later and enable TF32 if it does."""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        major, minor = capability

        if major >= 8:  # Ampere or newer
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            gpu_name = torch.cuda.get_device_name(device)
            logger.info(
                f"{gpu_name} (compute capability {major}.{minor}) supports TF32. Enabled for faster training."
            )


if __name__ == "__main__":
    # Check for TF32 support
    check_tf32_capability()
    main()
