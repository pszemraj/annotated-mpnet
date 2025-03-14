"""
Pretraining script for MPNet
"""

import argparse
import gc
import logging
import math
import os
import sys
from argparse import Namespace

from rich.logging import RichHandler

LOG_FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=LOG_FORMAT, datefmt="[%X] ", handlers=[RichHandler()]
)
LOGGER = logging.getLogger(__name__)


import torch
import torch.nn.functional as F
from datasets import load_dataset
from rich.progress import track
from torch.serialization import safe_globals
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from annotated_mpnet.data import (
    DataCollatorForMaskedPermutedLanguageModeling,
    HFStreamingDataset,
    MPNetDataset,
    RandomSamplerWithSeed,
)
from annotated_mpnet.modeling import MPNetForPretraining
from annotated_mpnet.scheduler import PolynomialDecayLRScheduler
from annotated_mpnet.tracking import AverageMeter
from annotated_mpnet.utils.utils import SUPPORTED_ACTIVATIONS


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
    The main function handling the training loop for MPNet pretraining
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

    # First test to see if max_positions and max_tokens are set differently. If they are, raise a
    # warning to the user to let them know this is very experimental and will most likely lead to
    # unexpect behavior
    if args.max_positions is not None:
        if args.max_positions != args.max_tokens:
            LOGGER.warning(
                "You have chosen to set a different number for max_positions and max_tokens. While "
                "this is allowed by this training script for experimental purposes, it will most "
                "likely lead to unexpected behavior. Please only proceed IF YOU KNOW WHAT YOU'RE "
                "DOING!!!"
            )

    # If max_positions is unset (as expected) we set max_positions to the same number as max_tokens
    # here
    if args.max_positions is None:
        args.max_positions = args.max_tokens

    # Now let's instantiate the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")

    # Check and adjust vocab_size parameter for better GPU performance
    original_vocab_size = tokenizer.vocab_size
    target_vocab_size = (
        (original_vocab_size + 127) // 128
    ) * 128  # Round up to nearest multiple of 128

    if target_vocab_size > original_vocab_size:
        LOGGER.info(
            f"Padding model's vocab_size from {original_vocab_size} to {target_vocab_size} "
            "(div. by 128) for GPU performance"
        )
        # Store both sizes in args for reference during conversion
        args.original_vocab_size = original_vocab_size
        args.padded_vocab_size = target_vocab_size
    else:
        args.original_vocab_size = original_vocab_size
        args.padded_vocab_size = original_vocab_size

    # Instantiate the tensorboard writers here
    if args.tensorboard_log_dir is not None:
        writers = {
            "train": SummaryWriter(os.path.join(args.tensorboard_log_dir, "train")),
            "valid": SummaryWriter(os.path.join(args.tensorboard_log_dir, "valid")),
            "test": SummaryWriter(os.path.join(args.tensorboard_log_dir, "test")),
        }

    # Next, we instantiate the model and the data collator
    model = MPNetForPretraining(args, tokenizer)
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
            # Load the dataset ONCE in streaming mode
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

            # First, create validation and test sets by taking samples
            LOGGER.info(
                f"Creating validation and test splits from streaming data (each with {args.eval_samples} samples)"
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

        # Load validation and test datasets from files (original code)
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

    # Finally, let's make sure the checkpoint directory exists. If not, let's create it
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # Before defining the scheduler and optimizer, let's make sure warmup_updates is set. If it
    # isn't, we need to set it to 10% the amount of total_updates
    if args.warmup_updates is None:
        args.warmup_updates = round(0.1 * args.total_updates)

    # Let's define an optimizer with our Polynomial decay scheduler on top of it
    # We set the optimizer with an arbitrary learning rate since it will be updated by the scheduler
    # anyway
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        betas=(args.beta1, args.beta2),
        lr=6e-9,  # starting learning rate during warmup
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        fused=True,
    )
    scheduler = PolynomialDecayLRScheduler(args, optimizer)

    # We create a step counter and an epoch counter here
    steps = 0
    epoch = 0

    # Create meters for all the relevant logging statistics using the Meters module
    meters = {
        "train_loss": AverageMeter(),
        "train_acc": AverageMeter(),
        "valid_loss": AverageMeter(),
        "valid_acc": AverageMeter(),
        "test_loss": AverageMeter(),
        "test_acc": AverageMeter(),
        "token_throughput": AverageMeter(),
    }

    # Additionally, we create a best loss counter that will be set arbitrarily high
    best_loss = 10e6

    while steps <= args.total_updates:
        # Handle either streaming or file-based training
        if train_streaming:
            LOGGER.info(f"Starting streaming training epoch {epoch}")

            # Create a single dataloader from our stream - no need to reload the dataset
            # Just shuffle with a different seed each epoch
            current_stream = train_stream.shuffle(
                buffer_size=args.buffer_size, seed=args.seed + epoch
            )

            train_dataloader = HFStreamingDataset(
                tokenizer=tokenizer,
                dataset_stream=current_stream,  # Use the already loaded stream
                block_size=args.max_tokens,
                buffer_size=args.buffer_size,
                seed=args.seed + epoch,  # Change seed each epoch for better variety
                text_field=args.text_field,
            )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataloader,
                batch_size=args.batch_size,
                collate_fn=mplm,
                num_workers=args.num_workers if hasattr(args, "num_workers") else 4,
            )
        else:
            # File-based datasets: Use a different file for each epoch (original code)
            current_train_file = train_files[epoch % len(train_files)]
            LOGGER.info(f"Training epoch {epoch} using file: {current_train_file}")

            # Load current file as dataset and set up dataloader
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

        # Zero out the gradients
        scheduler.optimizer.zero_grad()

        # Set the model in training mode to activate dropouts etc.
        model.train()

        # Create an accumulation loss and accumulation accuracy counter
        accumulation_loss = 0
        accumulation_acc = 0
        accumulation_tokens = 0

        # Create a counter that will keep track of total number of samples passed during a gradient
        # accumulation
        accumulation_sample_sizes = 0

        # Always reset the meters before beginning the next training epoch (except token throughput)
        for stat in ["train_loss", "train_acc", "valid_loss", "valid_acc"]:
            meters[stat].reset()

        # Now we do our training steps
        # We enumerate the dataloader because we need to keep track of gradient accumulation steps
        for i, batch in track(
            enumerate(train_dataloader),
            description=f"Training epoch {epoch}",
            total=len(train_dataloader) if not train_streaming else None,
        ):
            # Always check to make sure we haven't overstepped the total number of updates
            if steps > args.total_updates:
                break

            # Next check to see if we've hit a step interval and save that to the checkpoint dir
            if (
                (steps + 1) % args.checkpoint_interval == 0
                and args.checkpoint_interval > 0
                and steps > 0
            ):
                torch.save(
                    {"args": vars(args), "model_states": model.state_dict()},
                    os.path.join(args.checkpoint_dir, f"checkpoint{steps + 1}.pt"),
                )

            # Load the tensors onto the appropriate device
            device_batch = {
                data_type: (t.to(device) if isinstance(t, torch.Tensor) else t)
                for data_type, t in batch.items()
                if data_type != "attention_mask"
            }

            # Extract the targets since we'll use them a bunch below
            targets = device_batch["targets"]

            # Get the "sample_size" of the current batch, i.e., how many total targets there are to
            # be predicted. This will help us normalize the accumulated loss below
            accumulation_sample_sizes += targets.numel()

            # Update the count of total tokens processed during accumulation steps
            accumulation_tokens += device_batch["ntokens"]

            # Now let's process these through the model with autocast for mixed precision using bf16
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outs = model(**device_batch)

            # Process these out logits through cross entropy loss
            # Note: we do this outside of autocast to maintain precision for the loss calculation
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

            # Keep track of the accumulated accuracy to be divided by the total number of
            # accumulation steps before being written to tensorboard
            accumulation_acc += acc

            # Keep track of accumulated loss, to be divided by the total number of accumulation
            # steps later before being written to tensorboard
            accumulation_loss += loss.item()

            # Do the backward processing (but we won't step until we reach the gradient accumulation
            # number)
            loss.backward()

            # Check if we've reached a gradient accumulation step
            if (i + 1) % args.update_freq == 0:
                # Before stepping the optimizer, we need to normalize the gradients by the
                # accumulated sample sizes as described above
                if accumulation_sample_sizes > 0:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.data.mul_(1 / accumulation_sample_sizes)

                # We should also do a grad clip norm as well if it's been specified
                # If it hasn't been specified, we will calculate the gradient norm the old fashioned
                # way so that it can be logged
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

                # Now we step the scheduler (and return the LR so that we can store it)
                lr = scheduler.step(steps)

                # Reset gradients now
                scheduler.optimizer.zero_grad()

                # Calculate the accumulation normalized metrics by normalizing over the total number
                # of samples that have passed through each batch
                normal_acc = accumulation_acc / accumulation_sample_sizes
                normal_loss = (
                    accumulation_loss / accumulation_sample_sizes / math.log(2)
                )

                # Log some debugging values here
                LOGGER.debug("Accumulated batch information is below:")
                LOGGER.debug(accumulation_sample_sizes)
                LOGGER.debug(accumulation_loss)
                LOGGER.debug(accumulation_tokens)

                # Update the meters below
                meters["train_acc"].update(normal_acc, accumulation_sample_sizes)
                meters["train_loss"].update(normal_loss, accumulation_sample_sizes)
                meters["token_throughput"].update(accumulation_tokens)

                # Create a logging dict that will be passed to a tensorboard writer
                #
                # Quick rundown of the stats:
                # acc:
                #   model prediction accuracy, meter reset for each epoch
                # loss:
                #   loss output of the simulated batch (i.e. bsz times update_freq)
                # sbal:
                #   simulated batch averaged loss, keeping a running average of loss by simulated
                #   batch over the epoch (i.e. the meter is reset at the start of each epoch)
                # lr:
                #   keeping track of the learning rate
                # gnorm:
                #   keeping track of the norm of the gradient vector for the batch
                # ttp:
                #   total tokens processed, keeping a running count of the amount of training
                #   tokens the model has seen
                # tpb:
                #   tokens per batch, averaging out the tokens processed per batch
                logging_dict = {
                    "acc": meters["train_acc"].avg,
                    "loss": normal_loss,
                    "sbal": meters["train_loss"].avg,
                    "lr": lr,
                    "gnorm": gnorm,
                    "ttp": meters["token_throughput"].sum,
                    "tpb": meters["token_throughput"].avg,
                }

                # Now write to tensorboard
                if args.tensorboard_log_dir is not None:
                    write_to_tensorboard(writers["train"], logging_dict, steps)
                else:
                    LOGGER.info(logging_dict)

                # Reset accumulation counters here for the next set of accumulation steps
                accumulation_acc = 0
                accumulation_loss = 0
                accumulation_sample_sizes = 0
                accumulation_tokens = 0

                # Increment the step counter
                steps += 1

        # Set the model for validation
        model.eval()

        # Once the training loop is done, we begin the validation loop
        for i, batch in track(
            enumerate(valid_dataloader),
            description=f"Validation epoch {epoch}",
            total=len(valid_dataloader),
        ):
            # Load the tensors onto the appropriate device
            device_batch = {
                data_type: (t.to(device) if isinstance(t, torch.Tensor) else t)
                for data_type, t in batch.items()
                if data_type != "attention_mask"
            }

            # Extract the targets since we'll use them a bunch below
            targets = device_batch["targets"]

            # Now we move to no_grad since we don't have to calculate weights
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outs = model(**device_batch)

                # Calculate loss here (outside autocast for precision)
                loss = F.nll_loss(
                    F.log_softmax(
                        outs.view(-1, outs.size(-1)), dim=-1, dtype=torch.float32
                    ),
                    targets.view(-1),
                    reduction="sum",
                    ignore_index=tokenizer.pad_token_id,
                )

                normal_loss = loss.item() / targets.numel() / math.log(2)

                # Calculate accuracy here
                normal_acc = accuracy(outs, targets) / targets.view(-1).size(0)

                # Update the meters appropriately
                meters["valid_loss"].update(normal_loss, targets.numel())
                meters["valid_acc"].update(normal_acc, targets.numel())

        # Now we calculate post validation stats
        final_valid_loss = meters["valid_loss"].avg
        final_valid_accuracy = meters["valid_acc"].avg

        # Now let's save a best_val_checkpoint model
        if final_valid_loss < best_loss:
            # Reset the best loss to the new best loss for this epoch
            best_loss = final_valid_loss

            # Now let's go ahead and save this in the checkpoints directory
            torch.save(
                {"args": vars(args), "model_states": model.state_dict()},
                os.path.join(args.checkpoint_dir, "best_checkpoint.pt"),
            )

        # Load these into a logging dict and pass it on to be written in tensorboard
        logging_dict = {
            "loss": final_valid_loss,
            "acc": final_valid_accuracy,
            "best_loss": best_loss,
        }

        # Log to tensorboard or print out the dict
        if args.tensorboard_log_dir:
            write_to_tensorboard(writers["valid"], logging_dict, steps)
        else:
            LOGGER.info("Validation stats:")
            LOGGER.info(logging_dict)

        # Now, before looping back, we increment the epoch counter and we delete the train data
        # loader and garbage collect it
        epoch += 1

        # Clean up datasets if using file-based approach
        if not train_streaming:
            del train_dataloader
            del epoch_train_dataset
            gc.collect()
        else:
            del train_dataloader
            gc.collect()

    # If we've reached the end of the training cycle, i.e., hit total number of update steps, we can
    # use the test dataloader we built above to get a final test metric using the best checkpoint

    # Begin by loading the model states and args from the best checkpoint
    with safe_globals([Namespace]):
        dicts = torch.load(os.path.join(args.checkpoint_dir, "best_checkpoint.pt"))

    # Handle args that might be dict or Namespace
    loaded_args = dicts["args"]
    if isinstance(loaded_args, dict):
        loaded_args = Namespace(**loaded_args)

    # Handle potential _orig_mod prefix in state dict from compiled models
    model_states = dicts["model_states"]
    model_states = {k.replace("_orig_mod.", ""): v for k, v in model_states.items()}

    # Load an empty shell of the model architecture using those args
    test_model = MPNetForPretraining(loaded_args, tokenizer)

    # Now apply the model states to this newly instantiated model
    test_model.load_state_dict(model_states)

    # Finally make sure the model is in eval mode and is sent to the proper device
    test_model.to(device)
    test_model.eval()

    # Now we iterate through the test dataloader
    for i, batch in track(
        enumerate(test_dataloader),
        description="Test epoch",
        total=len(test_dataloader),
    ):
        # Load the tensors onto the appropriate device
        device_batch = {
            data_type: (t.to(device) if isinstance(t, torch.Tensor) else t)
            for data_type, t in batch.items()
            if data_type != "attention_mask"
        }

        # Extract the targets since we'll use them a bunch below
        targets = device_batch["targets"]

        # Now we move to no_grad since we don't have to calculate weights
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outs = test_model(**device_batch)

            # Calculate loss here (outside autocast for precision)
            loss = F.nll_loss(
                F.log_softmax(
                    outs.view(-1, outs.size(-1)), dim=-1, dtype=torch.float32
                ),
                targets.view(-1),
                reduction="sum",
                ignore_index=tokenizer.pad_token_id,
            )

            normal_loss = loss.item() / targets.numel() / math.log(2)

            # Calculate accuracy here
            normal_acc = accuracy(outs, targets) / targets.view(-1).size(0)

            # Update the meters appropriately
            meters["test_loss"].update(normal_loss, targets.numel())
            meters["test_acc"].update(normal_acc, targets.numel())

    # Now we calculate post test stats
    final_test_loss = meters["test_loss"].avg
    final_test_accuracy = meters["test_acc"].avg

    # Load these into a logging dict and pass it on to be written in tensorboard
    logging_dict = {
        "loss": final_test_loss,
        "acc": final_test_accuracy,
    }

    # Log to tensorboard or print out the dict
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
    Wrapper function so we can create a CLI entrypoint for this script
    """
    parser = argparse.ArgumentParser(
        description="Pretrain an MPNet model with a huggingface dataset "
        "or path(s) to local training/eval data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--encoder-layers",
        help="The number of encoder layers within the encoder block of MPNet. Defaults to 12, but "
        "can be increased for larger input sequences",
        default=12,
        type=int,
    )
    parser.add_argument(
        "--encoder-embed-dim",
        help="The dimension of the embedding layer inside each encoder block. Should generally "
        "always be 768, but some folks like OpenAI have seen good performance with large embeds",
        default=768,
        type=int,
    )
    parser.add_argument(
        "--encoder-ffn-dim",
        help="The dimension of the feed-forward hidden layer after each self-attention "
        "calculation. Defaults to 3072, but this can be any large number",
        default=3072,
        type=int,
    )
    parser.add_argument(
        "--encoder-attention-heads",
        help="The number of attention heads in each layer. Defaults to 12 which is what's used in"
        "bert-base and mpnet-base, but this can lend itself to some experimentation",
        default=12,
        type=int,
    )
    parser.add_argument(
        "--dropout",
        help="The standard dropout probability for the full encoder model. Defaults to 0.1",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--attention-dropout",
        help="The standard dropout probability for the attention layers of the encoder model. "
        "Defaults to 0.1",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--activation-dropout",
        help="The dropout probability after the activation function in the hidden layer of the "
        "feed-forward network after the attention calculation. Defaults to 0.1",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--max-positions",
        help="Max number of positional embeddings for the model. This should USUALLY always be the "
        "same number as the max sequence length (--max-tokens), but theoretically they could be "
        "different. However, this is not advised, so you should only set one of these",
        type=int,
    )
    parser.add_argument(
        "--max-tokens",
        help="Max number of tokens for input to the model. This should USUALLY always be the "
        "same number as the max positions (--max-positions), but theoretically they could be "
        "different. However, this is not advised, so you should only set one of these. Max tokens "
        "will default to 512",
        default=512,
        type=int,
    )
    parser.add_argument(
        "-activation",
        "--activation-fn",
        help="The activation function used throughout the model. Supported activations:\t"
        f"{', '.join(SUPPORTED_ACTIVATIONS)}",
        default="gelu",
        type=str,
    )
    parser.add_argument(
        "-prenorm",
        "--normalize-before",
        help="Determines when layer norm should be applied within each encoder layer.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--train-dir",
        help="The directory containing training files. This should generally be a directory since "
        "there are usually too many train files to fit in memory.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--valid-file",
        help="The file containing validation data.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--test-file",
        help="The file containing test data.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset-name",
        help="The name of the HuggingFace dataset to use (e.g., 'HuggingFaceFW/fineweb-edu'). If specified, this "
        "will override --train-dir, --valid-file, and --test-file.",
        type=str,
        default="gair-prox/DCLM-pro",
    )
    parser.add_argument(
        "--text-field",
        help="The field name in the dataset that contains the text to tokenize. Default is 'text'.",
        type=str,
        default="text",
    )
    parser.add_argument(
        "--buffer-size",
        help="Size of the buffer for streaming datasets. Larger buffers give better randomization but "
        "use more memory.",
        default=10000,
        type=int,
    )
    parser.add_argument(
        "--eval-samples",
        help="Number of samples to use for validation and test sets if they are not available in the "
        "dataset.",
        default=500,
        type=int,
    )
    parser.add_argument(
        "--min-text-length",
        help="Minimum text length to consider for examples from the dataset.",
        default=64,
        type=int,
    )
    parser.add_argument(
        "--total-updates",
        help="The maximum number of updates to do when training. Since we probably won't ever get "
        "through all the data, we need to set this",
        default=10000,
        type=int,
    )
    parser.add_argument(
        "-warmup_steps",
        "--warmup-updates",
        help="The number of warmup updates to increase the learning rate to --peak-lr before "
        "decreasing it again. Will default to 0.1 of --total-updates if left unset",
        type=int,
    )
    parser.add_argument(
        "--batch-size",
        help="The batch size to process at once. You can also use the --update-freq argument to do "
        "gradient accumulation if larger batches don't fit on the device",
        default=16,
        type=int,
    )
    parser.add_argument(
        "-gc_steps",
        "--update-freq",
        help="Amount of batches to process before updating the weights in the model, "
        "also known as gradient accumulation. Used to increase effective batch size without "
        "having to fit it all into memory.",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--beta1",
        help="The beta_1 of the Adam optimizer. Will default to 0.9",
        default=0.9,
        type=float,
    )
    parser.add_argument(
        "--beta2",
        help="The beta_2 of the Adam optimizer. Will default to 0.98",
        default=0.98,
        type=float,
    )
    parser.add_argument(
        "-wd",
        "--weight-decay",
        help="The weight decay for the AdamW optimizer.",
        default=0.01,
        type=float,
    )
    parser.add_argument(
        "-grad_clip",
        "--clip-grad-norm",
        help="The value above which to clip gradients down to.",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--lr",
        help="Peak learning rate that will be hit when the warmup updates have finished",
        default=6e-4,
        type=float,
    )
    parser.add_argument(
        "-end_lr",
        "--end-learning-rate",
        help="Target learning rate that the polynomial scheduler will slowly decrease to after warm-up.",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--adam-eps",
        help="The epsilon factor for the Adam optimizer that prevents divide by zero errors",
        default=1e-6,
        type=float,
    )
    parser.add_argument(
        "--power",
        help="The power of the polynomial decay of the LR scheduler",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "-save_steps",
        "--checkpoint-interval",
        help="The number of steps to be taken before saving the model",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--checkpoint-dir",
        help="The directory where model checkpoints are saved (inlucing 'best' and 'interval')",
        default="./checkpoints",
        type=str,
    )
    parser.add_argument(
        "-log_dir",
        "--tensorboard-log-dir",
        help="The directory to which tensorboard logs should be written. If this is unset, we will "
        "log stats to the terminal",
        type=str,
    )
    parser.add_argument(
        "--debug",
        help="Whether or not to output debug logs",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--seed",
        help="Set the random seed for training. Will default to 12345.",
        default=12345,
        type=int,
    )
    parser.add_argument(
        "--num-workers",
        help="Number of worker processes for data loading.",
        default=int(os.cpu_count() // 2),
        type=int,
    )
    parser.add_argument(
        "--compile",
        help="Whether or not to compile the model",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    # Check for validity of arguments
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
