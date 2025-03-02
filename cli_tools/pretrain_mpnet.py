#!/usr/bin/env python3
"""
Pretraining script for MPNet with FlexAttention support
This extends the original pretraining script with sliding window attention support.
"""

import logging
import sys

from rich.logging import RichHandler

LOG_FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=LOG_FORMAT, datefmt="[%X] ", handlers=[RichHandler()]
)
LOGGER = logging.getLogger(__name__)

import argparse
import gc
import math
import os

import torch
import torch.nn.functional as F
from datasets import load_dataset
from rich.progress import track
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from annotated_mpnet.data import (
    DataCollatorForMaskedPermutedLanguageModeling,
    HFStreamingDataset,
    MPNetDataset,
    RandomSamplerWithSeed,
)
from annotated_mpnet.modeling import MPNetFlexForPretraining
from annotated_mpnet.scheduler import PolynomialDecayLRScheduler
from annotated_mpnet.tracking import AverageMeter


def accuracy(output: torch.Tensor, target: torch.Tensor) -> int:
    """
    Helper function for comparing output logits to labels in target
    
    Args:
        output: the output logits of the model
        target: the labels generated from the collation process
        
    Returns:
        An accuracy prediction
    """
    with torch.no_grad():
        _, pred = output.topk(1, -1)
        correct = pred.view(-1).eq(target.view(-1))
    return correct.sum().item()


def write_to_tensorboard(writer: SummaryWriter, logging_dict: dict, step: int) -> None:
    """
    This function takes in a logging dict and sends it to tensorboard
    
    Args:
        writer: the SummaryWriter from tensorboard that writes the stats
        logging_dict: the dictionary containing the stats
        step: the current step
    """
    
    for stat_name, stat in logging_dict.items():
        writer.add_scalar(stat_name, stat, step)


def check_and_activate_tf32():
    """
    Check if the GPU supports NVIDIA Ampere or later and enable FP32 in PyTorch if it does.
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logging.info("No GPU detected, running on CPU.")
        return

    try:
        # Get the compute capability of the GPU
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        major, minor = capability

        # Check if the GPU is Ampere or newer (compute capability >= 8.0)
        if major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            gpu_name = torch.cuda.get_device_name(device)
            print(
                f"{gpu_name} (compute capability {major}.{minor}) supports NVIDIA Ampere or later, enabled TF32 in PyTorch."
            )
        else:
            gpu_name = torch.cuda.get_device_name(device)
            print(
                f"{gpu_name} (compute capability {major}.{minor}) does not support NVIDIA Ampere or later."
            )

    except Exception as e:
        logging.warning(f"Error occurred while checking GPU: {e}")


def main(args) -> None:
    """
    The main function handling the training loop for MPNet pretraining with FlexAttention
    """
    # Start by updating the LOGGER to run at debug level if the debug arg is true
    if args.debug:
        LOGGER.setLevel(logging.DEBUG)

    # Specify the torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        sys.exit(
            "CUDA is required for training MPNet. Please ensure that you have a CUDA enabled GPU."
        )

    check_and_activate_tf32()  # Check if the GPU supports NVIDIA Ampere or later and enable TF32

    # Check if sliding window size is set
    if args.sliding_window_size is not None:
        LOGGER.info(f"Using sliding window attention with window size: {args.sliding_window_size}")
    else:
        LOGGER.info("Using standard attention (no sliding window)")

    # First test to see if max_positions and max_tokens are set differently
    if args.max_positions is not None:
        if args.max_positions != args.max_tokens:
            LOGGER.warning(
                "You have chosen to set a different number for max_positions and max_tokens. While "
                "this is allowed by this training script for experimental purposes, it will most "
                "likely lead to unexpected behavior. Please only proceed IF YOU KNOW WHAT YOU'RE "
                "DOING!!!"
            )

    # If max_positions is unset, set max_positions to the same number as max_tokens
    if args.max_positions is None:
        args.max_positions = args.max_tokens

    # Now let's instantiate the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")

    # Instantiate the tensorboard writers here as well
    if args.tensorboard_log_dir is not None:
        writers = {
            "train": SummaryWriter(os.path.join(args.tensorboard_log_dir, "train")),
            "valid": SummaryWriter(os.path.join(args.tensorboard_log_dir, "valid")),
            "test": SummaryWriter(os.path.join(args.tensorboard_log_dir, "test")),
        }

    # Next, we instantiate the model and the data collator
    model = MPNetFlexForPretraining(args, tokenizer)
    mplm = DataCollatorForMaskedPermutedLanguageModeling(tokenizer=tokenizer)

    # Load the model up to the device
    model.to(device)

    # Compile the model
    if args.compile:
        LOGGER.info("Compiling the model...")
        model = torch.compile(model)

    # Determine whether to use streaming dataset or file-based dataset
    if args.dataset_name:
        LOGGER.info(f"Using HuggingFace dataset: {args.dataset_name}")

        try:
            # Load the dataset in streaming mode
            LOGGER.info(f"Loading streaming dataset: {args.dataset_name}")
            train_stream = load_dataset(
                args.dataset_name, split="train", streaming=True
            )

            # Apply minimum text length filter if specified
            if args.min_text_length > 0:
                train_stream = train_stream.filter(
                    lambda example: len(example[args.text_field])
                    >= args.min_text_length
                )

            # Create validation and test sets by taking samples
            LOGGER.info(
                f"Creating validation and test splits (each with {args.eval_samples} samples)"
            )

            # Take samples for validation
            valid_examples = []
            valid_iter = iter(train_stream.take(args.eval_samples))
            for _ in range(args.eval_samples):
                try:
                    valid_examples.append(next(valid_iter))
                except StopIteration:
                    LOGGER.warning(
                        f"Could only get {len(valid_examples)} examples for validation"
                    )
                    break

            # Take samples for test (skipping validation samples)
            test_examples = []
            test_iter = iter(
                train_stream.skip(args.eval_samples).take(args.eval_samples)
            )
            for _ in range(args.eval_samples):
                try:
                    test_examples.append(next(test_iter))
                except StopIteration:
                    LOGGER.warning(
                        f"Could only get {len(test_examples)} examples for testing"
                    )
                    break

            LOGGER.info(f"Created validation set with {len(valid_examples)} examples")
            LOGGER.info(f"Created test set with {len(test_examples)} examples")

            # Process validation and test examples
            valid_dataset = MPNetDataset(
                tokenizer=tokenizer,
                dataset=valid_examples,
                block_size=args.max_tokens,
                field_name=args.text_field,
            )

            test_dataset = MPNetDataset(
                tokenizer=tokenizer,
                dataset=test_examples,
                block_size=args.max_tokens,
                field_name=args.text_field,
            )

            # Create dataloaders
            valid_dataloader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=args.batch_size,
                collate_fn=mplm,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                collate_fn=mplm,
            )

            # Skip validation and test samples in the training stream
            train_stream = train_stream.skip(args.eval_samples * 2)
            train_streaming = True

        except Exception as e:
            LOGGER.error(f"Error loading dataset {args.dataset_name}: {e}")
            raise
    else:
        LOGGER.info("Using file-based datasets")
        train_streaming = False

        # Load validation and test datasets from files
        valid_dataset = MPNetDataset(
            tokenizer=tokenizer, file_path=args.valid_file, block_size=args.max_tokens
        )
        test_dataset = MPNetDataset(
            tokenizer=tokenizer, file_path=args.test_file, block_size=args.max_tokens
        )

        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, collate_fn=mplm, batch_size=args.batch_size
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, collate_fn=mplm, batch_size=args.batch_size
        )

        # Get each of the files in the training directory
        train_files = [
            f"{args.train_dir}{f}"
            for f in os.listdir(args.train_dir)
            if os.path.isfile(os.path.join(args.train_dir, f))
        ]

    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # Set warmup_updates if not specified (10% of total_updates)
    if args.warmup_updates is None:
        args.warmup_updates = round(0.1 * args.total_updates)

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        betas=(args.beta1, args.beta2),
        lr=6e-9,
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        fused=True,
    )
    scheduler = PolynomialDecayLRScheduler(args, optimizer)

    # Initialize counters
    steps = 0
    epoch = 0

    # Create meters for tracking statistics
    meters = {
        "train_loss": AverageMeter(),
        "train_acc": AverageMeter(),
        "valid_loss": AverageMeter(),
        "valid_acc": AverageMeter(),
        "test_loss": AverageMeter(),
        "test_acc": AverageMeter(),
        "token_throughput": AverageMeter(),
    }

    # Initialize best loss with a high value
    best_loss = 10e6

    # Main training loop
    while steps <= args.total_updates:
        # Handle either streaming or file-based training
        if train_streaming:
            LOGGER.info(f"Starting streaming training epoch {epoch}")

            # Create dataloader from stream
            current_stream = train_stream.shuffle(
                buffer_size=args.buffer_size, seed=args.seed + epoch
            )

            train_dataloader = HFStreamingDataset(
                tokenizer=tokenizer,
                dataset_stream=current_stream,
                block_size=args.max_tokens,
                buffer_size=args.buffer_size,
                seed=args.seed + epoch,
                text_field=args.text_field,
            )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataloader,
                batch_size=args.batch_size,
                collate_fn=mplm,
                num_workers=args.num_workers if hasattr(args, "num_workers") else 4,
            )
        else:
            # File-based datasets
            current_train_file = train_files[epoch % len(train_files)]
            LOGGER.info(f"Training epoch {epoch} using file: {current_train_file}")

            # Load current file as dataset
            epoch_train_dataset = MPNetDataset(
                tokenizer=tokenizer,
                file_path=current_train_file,
                block_size=args.max_tokens,
            )

            # Use seeded sampler for reproducibility
            sampler = RandomSamplerWithSeed(
                epoch_train_dataset, epoch=epoch, random_seed=args.seed
            )

            train_dataloader = torch.utils.data.DataLoader(
                epoch_train_dataset,
                sampler=sampler,
                collate_fn=mplm,
                batch_size=args.batch_size,
            )

        # Zero out gradients
        scheduler.optimizer.zero_grad()

        # Set model to training mode
        model.train()

        # Initialize accumulation counters
        accumulation_loss = 0
        accumulation_acc = 0
        accumulation_tokens = 0
        accumulation_sample_sizes = 0

        # Reset meters for new epoch
        for stat in ["train_loss", "train_acc", "valid_loss", "valid_acc"]:
            meters[stat].reset()

        # Training steps
        for i, batch in track(
            enumerate(train_dataloader),
            description=f"Training epoch {epoch}",
            total=len(train_dataloader) if not train_streaming else None,
        ):
            # Check if we've reached total updates
            if steps > args.total_updates:
                break

            # Save checkpoint at specified intervals
            if (
                (steps + 1) % args.checkpoint_interval == 0
                and args.checkpoint_interval > 0
                and steps > 0
            ):
                torch.save(
                    {"args": args, "model_states": model.state_dict()},
                    os.path.join(args.checkpoint_dir, f"checkpoint{steps + 1}.pt"),
                )

            # Move batch to device
            device_batch = {
                data_type: (t.to(device) if isinstance(t, torch.Tensor) else t)
                for data_type, t in batch.items()
                if data_type != "attention_mask"
            }

            # Extract targets
            targets = device_batch["targets"]

            # Update accumulation counters
            accumulation_sample_sizes += targets.numel()
            accumulation_tokens += device_batch["ntokens"]

            # Forward pass with autocast for mixed precision
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outs = model(
                    **device_batch, 
                    use_flex_attention=True  # Enable flex attention with sliding window
                )

            # Compute loss
            loss = F.nll_loss(
                F.log_softmax(
                    outs.view(-1, outs.size(-1)), dim=-1, dtype=torch.float32
                ),
                targets.view(-1),
                reduction="sum",
                ignore_index=tokenizer.pad_token_id,
            )

            # Calculate accuracy
            acc = accuracy(outs, targets)

            # Update accumulation counters
            accumulation_acc += acc
            accumulation_loss += loss.item()

            # Backward pass
            loss.backward()

            # Check if we've reached a gradient accumulation step
            if (i + 1) % args.update_freq == 0:
                # Normalize gradients
                if accumulation_sample_sizes > 0:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.data.mul_(1 / accumulation_sample_sizes)

                # Apply gradient clipping if specified
                if args.clip_grad_norm > 0.0:
                    gnorm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_grad_norm
                    )
                else:
                    gnorm = math.sqrt(
                        sum(
                            p.grad.data.norm() ** 2
                            for p in model.parameters()
                            if p.grad is not None
                        )
                    )

                # Step scheduler and optimizer
                lr = scheduler.step(steps)
                scheduler.optimizer.zero_grad()

                # Calculate normalized metrics
                normal_acc = accumulation_acc / accumulation_sample_sizes
                normal_loss = (
                    accumulation_loss / accumulation_sample_sizes / math.log(2)
                )

                # Debug output
                LOGGER.debug("Accumulated batch information is below:")
                LOGGER.debug(accumulation_sample_sizes)
                LOGGER.debug(accumulation_loss)
                LOGGER.debug(accumulation_tokens)

                # Update meters
                meters["train_acc"].update(normal_acc, accumulation_sample_sizes)
                meters["train_loss"].update(normal_loss, accumulation_sample_sizes)
                meters["token_throughput"].update(accumulation_tokens)

                # Create logging dict for tensorboard
                logging_dict = {
                    "acc": meters["train_acc"].avg,
                    "loss": normal_loss,
                    "sbal": meters["train_loss"].avg,
                    "lr": lr,
                    "gnorm": gnorm,
                    "ttp": meters["token_throughput"].sum,
                    "tpb": meters["token_throughput"].avg,
                }

                # Log to tensorboard or console
                if args.tensorboard_log_dir is not None:
                    write_to_tensorboard(writers["train"], logging_dict, steps)
                else:
                    LOGGER.info(logging_dict)

                # Reset accumulation counters
                accumulation_acc = 0
                accumulation_loss = 0
                accumulation_sample_sizes = 0
                accumulation_tokens = 0

                # Increment step counter
                steps += 1

        # Set model to evaluation mode for validation
        model.eval()

        # Validation loop
        for i, batch in track(
            enumerate(valid_dataloader),
            description=f"Validation epoch {epoch}",
            total=len(valid_dataloader),
        ):
            # Move batch to device
            device_batch = {
                data_type: (t.to(device) if isinstance(t, torch.Tensor) else t)
                for data_type, t in batch.items()
                if data_type != "attention_mask"
            }

            # Extract targets
            targets = device_batch["targets"]

            # Forward pass without gradient computation
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outs = model(
                        **device_batch,
                        use_flex_attention=True  # Enable flex attention with sliding window
                    )

                # Compute loss
                loss = F.nll_loss(
                    F.log_softmax(
                        outs.view(-1, outs.size(-1)), dim=-1, dtype=torch.float32
                    ),
                    targets.view(-1),
                    reduction="sum",
                    ignore_index=tokenizer.pad_token_id,
                )

                normal_loss = loss.item() / targets.numel() / math.log(2)

                # Calculate accuracy
                normal_acc = accuracy(outs, targets) / targets.view(-1).size(0)

                # Update meters
                meters["valid_loss"].update(normal_loss, targets.numel())
                meters["valid_acc"].update(normal_acc, targets.numel())

        # Calculate validation metrics
        final_valid_loss = meters["valid_loss"].avg
        final_valid_accuracy = meters["valid_acc"].avg

        # Save best model checkpoint
        if final_valid_loss < best_loss:
            best_loss = final_valid_loss
            torch.save(
                {"args": args, "model_states": model.state_dict()},
                os.path.join(args.checkpoint_dir, "best_checkpoint.pt"),
            )

        # Create logging dict for validation results
        logging_dict = {
            "loss": final_valid_loss,
            "acc": final_valid_accuracy,
            "best_loss": best_loss,
        }

        # Log validation results
        if args.tensorboard_log_dir:
            write_to_tensorboard(writers["valid"], logging_dict, steps)
        else:
            LOGGER.info("Validation stats:")
            LOGGER.info(logging_dict)

        # Increment epoch counter
        epoch += 1

        # Clean up datasets
        if not train_streaming:
            del train_dataloader
            del epoch_train_dataset
            gc.collect()
        else:
            del train_dataloader
            gc.collect()

    # Testing phase with best checkpoint
    LOGGER.info("Running final evaluation on test set with best checkpoint")
    
    # Load best checkpoint
    dicts = torch.load(os.path.join(args.checkpoint_dir, "best_checkpoint.pt"))
    test_model = MPNetFlexForPretraining(dicts["args"], tokenizer)
    test_model.load_state_dict(dicts["model_states"])
    test_model.to(device)
    test_model.eval()

    # Test loop
    for i, batch in track(
        enumerate(test_dataloader),
        description="Test evaluation",
        total=len(test_dataloader),
    ):
        # Move batch to device
        device_batch = {
            data_type: (t.to(device) if isinstance(t, torch.Tensor) else t)
            for data_type, t in batch.items()
            if data_type != "attention_mask"
        }

        # Extract targets
        targets = device_batch["targets"]

        # Forward pass without gradient computation
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outs = test_model(
                    **device_batch,
                    use_flex_attention=True  # Enable flex attention with sliding window
                )

            # Compute loss
            loss = F.nll_loss(
                F.log_softmax(
                    outs.view(-1, outs.size(-1)), dim=-1, dtype=torch.float32
                ),
                targets.view(-1),
                reduction="sum",
                ignore_index=tokenizer.pad_token_id,
            )

            normal_loss = loss.item() / targets.numel() / math.log(2)

            # Calculate accuracy
            normal_acc = accuracy(outs, targets) / targets.view(-1).size(0)

            # Update meters
            meters["test_loss"].update(normal_loss, targets.numel())
            meters["test_acc"].update(normal_acc, targets.numel())

    # Calculate test metrics
    final_test_loss = meters["test_loss"].avg
    final_test_accuracy = meters["test_acc"].avg

    # Create logging dict for test results
    logging_dict = {
        "loss": final_test_loss,
        "acc": final_test_accuracy,
    }

    # Log test results
    if args.tensorboard_log_dir:
        write_to_tensorboard(writers["test"], logging_dict, steps)
    else:
        LOGGER.info("Test stats:")
        LOGGER.info(logging_dict)

    LOGGER.info(
        f"Training is finished! See output in {args.checkpoint_dir} and "
        f"tensorboard logs in {args.tensorboard_log_dir}"
    )


def cli_main():
    """
    Wrapper function for the command-line interface
    """
    parser = argparse.ArgumentParser(
        description="Pretrain MPNet with FlexAttention",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model architecture arguments
    parser.add_argument(
        "--encoder-layers",
        help="The number of encoder layers",
        default=12,
        type=int,
    )
    parser.add_argument(
        "--encoder-embed-dim",
        help="The dimension of the embedding layer",
        default=768,
        type=int,
    )
    parser.add_argument(
        "--encoder-ffn-dim",
        help="The dimension of the feed-forward hidden layer",
        default=3072,
        type=int,
    )
    parser.add_argument(
        "--encoder-attention-heads",
        help="The number of attention heads in each layer",
        default=12,
        type=int,
    )
    
    # FlexAttention specific arguments
    parser.add_argument(
        "--sliding-window-size",
        help="Size of the sliding window for attention. None means full attention",
        default=None,
        type=int,
    )
    
    # Dropout arguments
    parser.add_argument(
        "--dropout",
        help="The standard dropout probability",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--attention-dropout",
        help="The dropout probability for attention layers",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--activation-dropout",
        help="The dropout probability after activation function",
        default=0.1,
        type=float,
    )
    
    # Position and token arguments
    parser.add_argument(
        "--max-positions",
        help="Max number of positional embeddings",
        type=int,
    )
    parser.add_argument(
        "--max-tokens",
        help="Max number of tokens for input",
        default=512,
        type=int,
    )
    
    # Activation and normalization arguments
    parser.add_argument(
        "--activation-fn",
        help="The activation function used throughout the model",
        default="gelu",
        type=str,
    )
    parser.add_argument(
        "--normalize-before",
        help="Whether to normalize before attention",
        action="store_true",
        default=False,
    )
    
    # Data source arguments
    parser.add_argument(
        "--train-dir",
        help="Directory containing training files",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--valid-file",
        help="File containing validation data",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--test-file",
        help="File containing test data",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset-name",
        help="HuggingFace dataset name (overrides file-based options)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--text-field",
        help="Field name in dataset containing text",
        type=str,
        default="text",
    )
    parser.add_argument(
        "--buffer-size",
        help="Size of buffer for streaming datasets",
        default=10000,
        type=int,
    )
    parser.add_argument(
        "--eval-samples",
        help="Number of samples for validation and test sets",
        default=500,
        type=int,
    )
    parser.add_argument(
        "--min-text-length",
        help="Minimum text length to consider",
        default=64,
        type=int,
    )
    
    # Training control arguments
    parser.add_argument(
        "--total-updates",
        help="Maximum number of updates for training",
        default=10000,
        type=int,
    )
    parser.add_argument(
        "--warmup-updates",
        help="Number of warmup updates",
        type=int,
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size for processing",
        default=16,
        type=int,
    )
    parser.add_argument(
        "-gc_steps",
        "--update-freq",
        help="Gradient accumulation steps",
        default=8,
        type=int,
    )
    
    # Optimizer arguments
    parser.add_argument(
        "--beta1",
        help="Beta_1 of Adam optimizer",
        default=0.9,
        type=float,
    )
    parser.add_argument(
        "--beta2",
        help="Beta_2 of Adam optimizer",
        default=0.98,
        type=float,
    )
    parser.add_argument(
        "--weight-decay",
        help="Weight decay for optimizer",
        default=0.01,
        type=float,
    )
    parser.add_argument(
        "-grad_clip",
        "--clip-grad-norm",
        help="Gradient clipping value",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--lr",
        help="Peak learning rate",
        default=0.0002,
        type=float,
    )
    parser.add_argument(
        "-end_lr",
        "--end-learning-rate",
        help="Final learning rate",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--adam-eps",
        help="Epsilon for Adam optimizer",
        default=1e-6,
        type=float,
    )
    parser.add_argument(
        "--power",
        help="Power of polynomial decay for scheduler",
        default=1.0,
        type=float,
    )
    
    # Checkpoint and logging arguments
    parser.add_argument(
        "-save_steps",
        "--checkpoint-interval",
        help="Steps between model checkpoints",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--checkpoint-dir",
        help="Directory for saving checkpoints",
        default="./checkpoints",
        type=str,
    )
    parser.add_argument(
        "-log_dir",
        "--tensorboard-log-dir",
        help="Directory for tensorboard logs",
        type=str,
    )
    
    # Other arguments
    parser.add_argument(
        "--debug",
        help="Enable debug logging",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--seed",
        help="Random seed for training",
        default=12345,
        type=int,
    )
    parser.add_argument(
        "--num-workers",
        help="Number of worker processes for data loading",
        default=int(os.cpu_count() // 2),
        type=int,
    )
    parser.add_argument(
        "--compile",
        help="Whether to compile the model",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    # Validate arguments
    if args.dataset_name is None and (
        args.train_dir is None or args.valid_file is None or args.test_file is None
    ):
        parser.error(
            "Either --dataset-name or (--train-dir, --valid-file, and --test-file) must be provided."
        )

    LOGGER.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()