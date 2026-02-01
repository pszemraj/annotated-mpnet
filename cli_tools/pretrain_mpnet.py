"""
Pretraining script for MPNet
"""

import argparse
import contextlib
import gc
import json
import logging
import math
import os
import pathlib
import sys
from argparse import Namespace

import numpy as np
from rich.logging import RichHandler

LOG_FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=LOG_FORMAT, datefmt="[%X] ", handlers=[RichHandler()])
LOGGER = logging.getLogger(__name__)


import torch
import torch.nn.functional as F
import wandb
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
from annotated_mpnet.utils.utils import (
    SUPPORTED_ACTIVATIONS,
    model_summary,
    validate_tokenizer,
)

DEFAULT_BEST_LOSS = 10e6


def accuracy(output: torch.Tensor, target: torch.Tensor, ignore_index: int | None = None) -> int:
    """Compare output logits to labels and return correct predictions.

    :param torch.Tensor output: Output logits of the model.
    :param torch.Tensor target: Labels generated from the collation process.
    :param int ignore_index: Token ID to ignore in accuracy, defaults to None.
    :return int: Number of correct predictions.
    """
    with torch.no_grad():
        _, pred = output.topk(1, -1)
        pred = pred.view(-1)
        target = target.view(-1)
        if ignore_index is not None:
            mask = target.ne(ignore_index)
            pred = pred[mask]
            target = target[mask]
        correct = pred.eq(target)
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


def log_to_wandb(logging_dict: dict, step: int, split: str) -> None:
    """
    Log metrics to Weights & Biases

    Args:
        logging_dict: the dictionary containing the stats
        step: the current step
        split: the data split (train, valid, test)
    """
    if wandb.run is not None:
        # Prefix metrics with split name for better organization in the dashboard
        wandb_dict = {f"{split}/{k}": v for k, v in logging_dict.items()}
        wandb_dict["step"] = step
        wandb.log(wandb_dict)


def _get_initial_best_loss(checkpoint: dict | None) -> float:
    """Return the best loss from a checkpoint or a default value.

    :param dict checkpoint: Loaded checkpoint or None.
    :return float: Best loss value.
    """
    if checkpoint is None:
        return DEFAULT_BEST_LOSS

    return checkpoint.get("best_loss", DEFAULT_BEST_LOSS)


def _warn_if_max_positions_mismatch(args: Namespace) -> None:
    """Warn if max_positions and max_tokens are set to different values.

    :param Namespace args: Parsed CLI arguments.
    :return None: This function returns nothing.
    """
    if args.max_positions is not None and args.max_positions != args.max_tokens:
        LOGGER.warning(
            "You have chosen to set a different number for max_positions and max_tokens. While "
            "this is allowed by this training script for experimental purposes, it will most "
            "likely lead to unexpected behavior. Please only proceed IF YOU KNOW WHAT YOU'RE "
            "DOING!!!"
        )


def _apply_checkpoint_architecture_args(args: Namespace, checkpoint_args: Namespace | dict) -> None:
    """Apply architecture settings from a checkpoint to the args.

    :param Namespace args: Current CLI args.
    :param Namespace | dict checkpoint_args: Stored checkpoint args.
    :return None: This function returns nothing.
    """
    if isinstance(checkpoint_args, Namespace):
        checkpoint_args = vars(checkpoint_args)

    # Restore model architecture parameters
    args.encoder_layers = checkpoint_args["encoder_layers"]
    args.encoder_embed_dim = checkpoint_args["encoder_embed_dim"]
    args.encoder_ffn_dim = checkpoint_args["encoder_ffn_dim"]
    args.encoder_attention_heads = checkpoint_args["encoder_attention_heads"]
    args.dropout = checkpoint_args.get("dropout", args.dropout)
    args.attention_dropout = checkpoint_args.get("attention_dropout", args.attention_dropout)
    args.activation_dropout = checkpoint_args.get("activation_dropout", args.activation_dropout)
    args.activation_fn = checkpoint_args.get("activation_fn", args.activation_fn)
    args.relative_attention_num_buckets = checkpoint_args.get(
        "relative_attention_num_buckets", args.relative_attention_num_buckets
    )
    args.original_vocab_size = checkpoint_args.get("original_vocab_size", args.original_vocab_size)
    args.padded_vocab_size = checkpoint_args.get("padded_vocab_size", args.padded_vocab_size)

    args.max_tokens = checkpoint_args.get("max_tokens", args.max_tokens)
    if "max_positions" in checkpoint_args:
        args.max_positions = checkpoint_args.get("max_positions", args.max_positions)
    else:
        args.max_positions = args.max_tokens


def _select_architecture_source(args: Namespace) -> str:
    """Select the architecture source based on CLI arguments.

    :param Namespace args: Parsed CLI arguments.
    :return str: One of "hf", "resume", or "new".
    """
    if args.hf_model_path is not None:
        return "hf"
    if args.resume:
        return "resume"
    return "new"


def _select_resume_checkpoint_path(
    checkpoint_dir: pathlib.Path, resume_checkpoint: str | None
) -> pathlib.Path:
    """Select the checkpoint path for resuming training.

    :param pathlib.Path checkpoint_dir: Base checkpoint directory.
    :param str resume_checkpoint: Explicit checkpoint path or None.
    :return pathlib.Path: Checkpoint path to resume from.
    """
    if resume_checkpoint is None:
        return checkpoint_dir / "best_checkpoint.pt"
    return pathlib.Path(resume_checkpoint)


def _select_optimizer_state_path(
    optimizer_dir: pathlib.Path, resume_checkpoint_path: pathlib.Path
) -> pathlib.Path:
    """Select the optimizer state path that matches the resume checkpoint.

    :param pathlib.Path optimizer_dir: Directory containing optimizer states.
    :param pathlib.Path resume_checkpoint_path: Checkpoint being resumed.
    :return pathlib.Path: Optimizer state path to load.
    """
    if resume_checkpoint_path.name == "best_checkpoint.pt":
        return optimizer_dir / "best_optimizer_state.pt"
    return optimizer_dir / f"{resume_checkpoint_path.stem}_optimizer_state.pt"


def _normalize_training_accuracy(accumulation_acc: float, accumulation_pred_tokens: int) -> float:
    """Normalize accumulated accuracy by predicted token count.

    :param float accumulation_acc: Accumulated correct predictions.
    :param int accumulation_pred_tokens: Number of predicted tokens.
    :return float: Normalized accuracy value.
    """
    if accumulation_pred_tokens == 0:
        return 0.0
    return accumulation_acc / accumulation_pred_tokens


def _count_pred_tokens(targets: torch.Tensor, pad_token_id: int) -> int:
    """Count the number of non-pad tokens in targets.

    :param torch.Tensor targets: Target token IDs.
    :param int pad_token_id: Padding token ID to ignore.
    :return int: Number of non-pad tokens.
    """
    return int(targets.ne(pad_token_id).sum().item())


def _get_autocast_context(device: torch.device) -> contextlib.AbstractContextManager[None]:
    """Return the autocast context manager for the given device.

    :param torch.device device: Device used for training/inference.
    :return contextlib.AbstractContextManager[None]: Autocast context manager.
    """
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _scale_gradients_by_tokens(model: torch.nn.Module, total_tokens: int) -> None:
    """Scale gradients by the total number of tokens.

    :param torch.nn.Module model: Model with accumulated gradients.
    :param int total_tokens: Total token count for normalization.
    :return None: This function returns nothing.
    """
    if total_tokens <= 0:
        return
    for param in model.parameters():
        if param.grad is not None:
            param.grad.div_(total_tokens)


def check_and_activate_tf32() -> None:
    """Check GPU capability and enable TF32 if supported.

    :return None: This function returns nothing.
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


def main(args: Namespace) -> None:
    """Run the MPNet pretraining loop.

    :param Namespace args: Parsed CLI arguments.
    :return None: This function returns nothing.
    """
    # Start by updating the LOGGER to run at debug level if the debug arg is true
    if args.debug:
        LOGGER.setLevel(logging.DEBUG)

    # Check the torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and os.getenv("MPNET_CPU_OVERRIDE", "0") != "1":
        sys.exit(
            "CUDA is required for training MPNet. Please ensure that you have a CUDA enabled GPU."
        )

    check_and_activate_tf32()  # Check if the GPU supports NVIDIA Ampere or later and enable TF32

    # First test to see if max_positions and max_tokens are set differently. If they are, raise a
    # warning to the user to let them know this is very experimental and will most likely lead to
    # unexpect behavior
    _warn_if_max_positions_mismatch(args)

    # If max_positions is unset (as expected) we set max_positions to the same number as max_tokens
    if args.max_positions is None:
        args.max_positions = args.max_tokens

    # TOKENIZER
    # -----------------------------------

    LOGGER.info(f"Loading tokenizer from {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, model_max_length=args.max_tokens)
    is_valid, details = validate_tokenizer(tokenizer)
    assert is_valid and details["whole_word_mask"], (
        f"Invalid tokenizer: {args.tokenizer_name}. Debug w/ verbose output from validate_tokenizer()"
    )

    # Get the tokenizer vocab size
    original_vocab_size = len(
        tokenizer
    )  # Use len() to get actual vocab size including added tokens

    # Determine whether to pad vocab size based on whether we're loading from existing model
    if args.resume or args.hf_model_path is not None:
        # When loading from existing model, use its vocab size (will be set later from checkpoint/HF config)
        LOGGER.info("Loading from existing model - will use model's vocab size")
        # These will be overridden when we load the checkpoint or HF config
        args.original_vocab_size = original_vocab_size
        args.padded_vocab_size = original_vocab_size
    else:
        # When training from scratch, pad vocab size for GPU performance
        target_vocab_size = ((original_vocab_size + 127) // 128) * 128  # Round up to nearest 128

        if target_vocab_size > original_vocab_size:
            LOGGER.info(
                f"Training from scratch - padding vocab_size from {original_vocab_size} to {target_vocab_size} "
                "(div. by 128) for GPU performance"
            )
            args.original_vocab_size = original_vocab_size
            args.padded_vocab_size = target_vocab_size
        else:
            LOGGER.info(f"Using tokenizer vocab_size: {original_vocab_size}")
            args.original_vocab_size = original_vocab_size
            args.padded_vocab_size = original_vocab_size

    # Explicitly store token IDs in args for consistent usage
    args.pad_token_id = tokenizer.pad_token_id
    args.bos_token_id = tokenizer.bos_token_id
    args.eos_token_id = tokenizer.eos_token_id

    # -----------------------------------

    # Instantiate the tensorboard writers
    if args.tensorboard_log_dir is not None:
        writers = {
            "train": SummaryWriter(os.path.join(args.tensorboard_log_dir, "train")),
            "valid": SummaryWriter(os.path.join(args.tensorboard_log_dir, "valid")),
            "test": SummaryWriter(os.path.join(args.tensorboard_log_dir, "test")),
        }

    # Check if we're resuming and need to load architecture from checkpoint
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    arch_source = _select_architecture_source(args)

    if arch_source == "resume":
        # Load checkpoint to get architecture before creating model
        resume_checkpoint_path = _select_resume_checkpoint_path(
            checkpoint_dir, args.resume_checkpoint
        )

        LOGGER.info(f"Loading architecture from checkpoint: {resume_checkpoint_path}")
        with safe_globals([Namespace, np.ndarray, np.dtype, np._core.multiarray._reconstruct]):
            resume_checkpoint = torch.load(
                resume_checkpoint_path, map_location="cpu", weights_only=False
            )

        # Restore architecture args from checkpoint
        if "args" in resume_checkpoint:
            checkpoint_args = resume_checkpoint["args"]
            _apply_checkpoint_architecture_args(args, checkpoint_args)
            tokenizer.model_max_length = args.max_tokens
            _warn_if_max_positions_mismatch(args)

            LOGGER.info(
                f"Restored model architecture from checkpoint: {args.encoder_layers} layers, "
                f"{args.encoder_embed_dim} hidden, {args.encoder_ffn_dim} FFN"
            )

    # If loading from HuggingFace model, we need to get the config first
    elif arch_source == "hf":
        LOGGER.info(f"Loading config from HuggingFace model: {args.hf_model_path}")
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(args.hf_model_path)

        # Override args with HF model's architecture
        args.encoder_layers = hf_config.num_hidden_layers
        args.encoder_embed_dim = hf_config.hidden_size
        args.encoder_ffn_dim = hf_config.intermediate_size
        args.encoder_attention_heads = hf_config.num_attention_heads
        args.dropout = hf_config.hidden_dropout_prob
        args.attention_dropout = hf_config.attention_probs_dropout_prob
        args.activation_dropout = hf_config.hidden_dropout_prob
        args.activation_fn = hf_config.hidden_act
        # Set max_positions from HF config (already includes special tokens)
        args.max_positions = hf_config.max_position_embeddings
        args.relative_attention_num_buckets = getattr(
            hf_config, "relative_attention_num_buckets", 32
        )
        args.original_vocab_size = hf_config.vocab_size
        args.padded_vocab_size = hf_config.vocab_size

        LOGGER.info(
            f"Using HF model architecture: {args.encoder_layers} layers, "
            f"{args.encoder_embed_dim} hidden, {args.encoder_ffn_dim} FFN"
        )

    # Next, we instantiate the model and the data collator
    model = MPNetForPretraining(args, tokenizer)
    mplm = DataCollatorForMaskedPermutedLanguageModeling(tokenizer=tokenizer)

    # Initialize wandb if enabled (after model creation)
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            resume="allow",
            id=args.wandb_id,
        )
        # Log model architecture as a graph
        if args.wandb_watch:
            wandb.watch(model, log_freq=100)

    model_summary(model, max_depth=3)

    # sync args for relative attention with model
    args.relative_attention_num_buckets = model.sentence_encoder.relative_attention_num_buckets
    args.relative_attention_max_distance = model.sentence_encoder.relative_attention_max_distance

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
            train_stream = load_dataset(args.dataset_name, split="train", streaming=True)

            # Apply minimum text length filter if specified
            if args.min_text_length > 0:
                train_stream = train_stream.filter(
                    lambda example: len(example[args.text_field]) >= args.min_text_length
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
                    LOGGER.warning(f"Could only get {len(valid_examples)} examples for validation")
                    break

            # Take samples for test (skipping validation samples)
            test_examples = []
            test_iter = iter(train_stream.skip(args.eval_samples).take(args.eval_samples))
            for _ in range(args.eval_samples):
                try:
                    test_examples.append(next(test_iter))
                except StopIteration:
                    LOGGER.warning(f"Could only get {len(test_examples)} examples for testing")
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

    # Note: checkpoint_dir is already created above when handling resume logic

    # Create optimizer state directory if saving optimizer states
    if args.save_optimizer_state:
        optimizer_dir = checkpoint_dir / "optimizer"
        optimizer_dir.mkdir(parents=True, exist_ok=True)

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
        fused=device.type == "cuda",
    )
    scheduler = PolynomialDecayLRScheduler(args, optimizer)

    # Initialize step and epoch counters
    steps = 0
    epoch = 0
    best_loss = DEFAULT_BEST_LOSS  # Will be overridden if resuming from checkpoint

    # Determine whether to load from HuggingFace model or resume from a checkpoint
    # HuggingFace model loading takes precedence if both are specified
    if args.hf_model_path is not None:
        LOGGER.info(f"Initializing model from HuggingFace model: {args.hf_model_path}")

        try:
            LOGGER.info(f"Checking if HuggingFace model path exists: {args.hf_model_path}")

            # Import the converter
            from cli_tools.convert_hf_model_to_mpnet import convert_hf_model_to_mpnet

            # Create a temporary checkpoint path for the converted model
            temp_checkpoint_path = checkpoint_dir / "hf_converted_checkpoint.pt"
            LOGGER.info(f"Will save converted model to: {temp_checkpoint_path}")

            # Convert the HuggingFace model to our format
            try:
                convert_hf_model_to_mpnet(
                    args.hf_model_path,
                    str(temp_checkpoint_path),
                )
            except Exception as conv_error:
                LOGGER.error(f"Error during model conversion: {conv_error}")
                LOGGER.error("When --hf-model-path is specified, model loading MUST succeed.")
                LOGGER.error(
                    "To use default random initialization, remove the --hf-model-path argument."
                )
                raise RuntimeError(f"Failed to load HuggingFace model: {conv_error}")

            # Now load the converted checkpoint
            LOGGER.info(f"Loading converted checkpoint from {temp_checkpoint_path}")
            if not temp_checkpoint_path.exists():
                raise FileNotFoundError(f"Converted checkpoint not found at {temp_checkpoint_path}")

            with safe_globals([Namespace, np.ndarray, np.dtype, np._core.multiarray._reconstruct]):
                checkpoint = torch.load(
                    temp_checkpoint_path, map_location=device, weights_only=False
                )

            # Extract model states
            model_states = checkpoint["model_states"]
            model_states = {k.replace("_orig_mod.", ""): v for k, v in model_states.items()}

            # Load model weights
            model.load_state_dict(model_states)
            LOGGER.info("Model weights loaded successfully from HuggingFace model")

        except Exception as e:
            LOGGER.error(f"Error loading HuggingFace model: {e}")
            LOGGER.warning(f"Full error: {str(e)}")
            LOGGER.warning("Proceeding with default initialization")

    # Handle resuming from checkpoint if enabled
    elif args.resume:
        # Note: We already loaded the checkpoint above to get architecture
        # Now we just need to extract the other state
        checkpoint = resume_checkpoint

        # Extract training state
        if "steps" in checkpoint:
            steps = checkpoint["steps"]
            LOGGER.info(f"Resuming from step {steps}")

        if "epoch" in checkpoint:
            epoch = checkpoint["epoch"]
            LOGGER.info(f"Resuming from epoch {epoch}")

        best_loss = _get_initial_best_loss(checkpoint)
        if "best_loss" in checkpoint:
            LOGGER.info(f"Best validation loss from checkpoint: {best_loss}")

        # Restore RNG state if available
        if "rng_state" in checkpoint:
            try:
                rng_state = checkpoint["rng_state"]
                if rng_state.get("torch") is not None:
                    # Ensure the RNG state is a ByteTensor
                    torch_rng = rng_state["torch"]
                    if not isinstance(torch_rng, torch.ByteTensor):
                        if hasattr(torch_rng, "cpu"):
                            torch_rng = torch_rng.cpu()
                        torch_rng = torch.ByteTensor(torch_rng)
                    torch.set_rng_state(torch_rng)
                    LOGGER.info("Restored torch RNG state")

                if torch.cuda.is_available() and rng_state.get("cuda") is not None:
                    cuda_states = rng_state["cuda"]
                    # Handle list of CUDA states for multi-GPU
                    if isinstance(cuda_states, list):
                        processed_states = []
                        for state in cuda_states:
                            if not isinstance(state, torch.ByteTensor):
                                if hasattr(state, "cpu"):
                                    state = state.cpu()
                                state = torch.ByteTensor(state)
                            processed_states.append(state)
                        torch.cuda.set_rng_state_all(processed_states)
                    else:
                        if not isinstance(cuda_states, torch.ByteTensor):
                            if hasattr(cuda_states, "cpu"):
                                cuda_states = cuda_states.cpu()
                            cuda_states = torch.ByteTensor(cuda_states)
                        torch.cuda.set_rng_state(cuda_states)
                    LOGGER.info("Restored CUDA RNG state")

                if "numpy.random" in sys.modules and rng_state.get("numpy") is not None:
                    np.random.set_state(rng_state["numpy"])
                    LOGGER.info("Restored numpy RNG state")
            except (TypeError, ValueError, AttributeError) as e:
                LOGGER.warning(f"Could not restore RNG state: {e}")
                LOGGER.info("Continuing with current RNG state")

        # Extract model states
        model_states = checkpoint["model_states"]
        model_states = {k.replace("_orig_mod.", ""): v for k, v in model_states.items()}

        # Load model weights
        model.load_state_dict(model_states)
        LOGGER.info("Model weights loaded successfully")

        # Check if optimizer state exists and load it if requested
        if args.save_optimizer_state:
            optimizer_state_path = _select_optimizer_state_path(
                optimizer_dir, resume_checkpoint_path
            )
            if optimizer_state_path.exists():
                LOGGER.info(f"Loading optimizer state from {optimizer_state_path}")
                with safe_globals(
                    [Namespace, np.ndarray, np.dtype, np._core.multiarray._reconstruct]
                ):
                    optimizer_state = torch.load(
                        optimizer_state_path, map_location=device, weights_only=False
                    )

                # Load optimizer state
                optimizer.load_state_dict(optimizer_state["optimizer"])

                # Load scheduler state
                scheduler.load_state_dict(optimizer_state["scheduler"])

                LOGGER.info("Optimizer and scheduler states loaded successfully")
            else:
                LOGGER.warning(
                    f"No optimizer state found at {optimizer_state_path}, using default initialization"
                )

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

    # best_loss is already initialized above (possibly from a checkpoint)
    # Flag to track if non-trainable model repo files were saved at first checkpoint.
    initial_outputs_saved = False

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
            sampler = RandomSamplerWithSeed(epoch_train_dataset, epoch=epoch, random_seed=args.seed)

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
        accumulation_input_tokens = 0
        accumulation_pred_tokens = 0

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
                # Create checkpoint object with model state and args
                checkpoint = {
                    "args": vars(args),
                    "model_states": model.state_dict(),
                    "steps": steps,
                    "epoch": epoch,
                    "rng_state": {
                        "torch": torch.get_rng_state(),
                        "cuda": (
                            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                        ),
                        "numpy": (np.random.get_state() if "numpy.random" in sys.modules else None),
                    },
                }

                # Save the checkpoint
                checkpoint_path = checkpoint_dir / f"checkpoint{steps + 1}.pt"
                torch.save(checkpoint, checkpoint_path)

                # If optimizer state saving is enabled, save the optimizer state to the optimizer directory
                if args.save_optimizer_state:
                    optimizer_state = {
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "steps": steps,
                        "epoch": epoch,
                    }
                    optimizer_state_path = (
                        optimizer_dir / f"{checkpoint_path.stem}_optimizer_state.pt"
                    )
                    torch.save(optimizer_state, optimizer_state_path)

                # Save the args & tokenizer if this is the first checkpoint
                if not initial_outputs_saved:
                    args_dict = vars(args) if not isinstance(args, dict) else args
                    args_path = checkpoint_dir / "training_args.json"
                    with open(args_path, "w") as f:
                        json.dump(args_dict, f, indent=4)

                    tokenizer_dir = checkpoint_dir / "tokenizer"
                    tokenizer.save_pretrained(tokenizer_dir)
                    initial_outputs_saved = True

            # Load the tensors onto the appropriate device
            device_batch = {
                data_type: (t.to(device) if isinstance(t, torch.Tensor) else t)
                for data_type, t in batch.items()
                if data_type != "attention_mask"
            }

            # Extract the targets since we'll use them a bunch below
            targets = device_batch["targets"]

            # Track batch statistics for metrics
            input_token_count = int(device_batch["ntokens"])
            pred_token_count = _count_pred_tokens(targets, tokenizer.pad_token_id)

            # Add to accumulation counters for later metric calculations
            accumulation_input_tokens += input_token_count
            accumulation_pred_tokens += pred_token_count

            # Now let's process these through the model with autocast for mixed precision using bf16
            with _get_autocast_context(device):
                outs = model(**device_batch)

            # Process these out logits through cross entropy loss
            # Note: we do this outside of autocast to maintain precision for the loss calculation
            loss = F.nll_loss(
                F.log_softmax(outs.view(-1, outs.size(-1)), dim=-1, dtype=torch.float32),
                targets.view(-1),
                reduction="sum",
                ignore_index=tokenizer.pad_token_id,
            )

            # Calculate accuracy
            acc = accuracy(outs, targets, ignore_index=tokenizer.pad_token_id)

            # Accumulate loss and accuracy stats
            accumulation_acc += acc
            accumulation_loss += loss.item()

            # Accumulate raw gradients; normalize at the accumulation boundary
            loss.backward()

            # Check if we've reached a gradient accumulation step
            if (i + 1) % args.update_freq == 0:
                # Scale accumulated gradients by total predicted tokens for correct GA
                _scale_gradients_by_tokens(model, accumulation_pred_tokens)

                # Apply gradient clipping on the accumulated gradients
                if args.clip_grad_norm > 0.0:
                    gnorm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_grad_norm
                    ).item()
                else:
                    gnorm = math.sqrt(
                        sum(
                            p.grad.data.norm() ** 2
                            for p in model.parameters()
                            if p.grad is not None
                        )
                    )  # record gradient norm for logging

                # Now we step the scheduler (and return the LR so that we can store it)
                lr = scheduler.step(steps)

                # Reset gradients now
                scheduler.optimizer.zero_grad()

                # Calculate metrics - since we're now tracking per-token loss, our normalization is simpler
                # We just need to average across the accumulated steps
                # For accuracy, normalize by the total number of predicted tokens
                normal_acc = _normalize_training_accuracy(
                    accumulation_acc, accumulation_pred_tokens
                )
                # Convert to bits per token using true token normalization
                if accumulation_pred_tokens > 0:
                    normal_loss = accumulation_loss / accumulation_pred_tokens / math.log(2)
                else:
                    normal_loss = 0.0

                # Log some debugging values here
                LOGGER.debug("Accumulated batch information is below:")
                LOGGER.debug(accumulation_pred_tokens)
                LOGGER.debug(accumulation_loss)
                LOGGER.debug(accumulation_input_tokens)

                # Update the meters below
                meters["train_acc"].update(normal_acc, accumulation_pred_tokens)
                meters["train_loss"].update(normal_loss, accumulation_pred_tokens)
                meters["token_throughput"].update(accumulation_input_tokens)

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

                # Log to wandb if enabled
                if args.wandb:
                    log_to_wandb(logging_dict, steps, "train")

                # Reset accumulation counters here for the next set of accumulation steps
                accumulation_acc = 0
                accumulation_loss = 0
                accumulation_input_tokens = 0
                accumulation_pred_tokens = 0

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
                with _get_autocast_context(device):
                    outs = model(**device_batch)

                # Calculate loss here (outside autocast for precision)
                loss = F.nll_loss(
                    F.log_softmax(outs.view(-1, outs.size(-1)), dim=-1, dtype=torch.float32),
                    targets.view(-1),
                    reduction="sum",
                    ignore_index=tokenizer.pad_token_id,
                )

                pred_tokens = _count_pred_tokens(targets, tokenizer.pad_token_id)
                if pred_tokens > 0:
                    normal_loss = loss.item() / pred_tokens / math.log(2)
                    normal_acc = (
                        accuracy(outs, targets, ignore_index=tokenizer.pad_token_id) / pred_tokens
                    )
                    # Update the meters appropriately
                    meters["valid_loss"].update(normal_loss, pred_tokens)
                    meters["valid_acc"].update(normal_acc, pred_tokens)

        # Now we calculate post validation stats
        final_valid_loss = meters["valid_loss"].avg
        final_valid_accuracy = meters["valid_acc"].avg

        # Now let's save a best_val_checkpoint model
        if final_valid_loss < best_loss:
            # Reset the best loss to the new best loss for this epoch
            best_loss = final_valid_loss

            # Create best checkpoint object with model state and args
            best_checkpoint = {
                "args": vars(args),
                "model_states": model.state_dict(),
                "steps": steps,
                "epoch": epoch,
                "best_loss": best_loss,
                "rng_state": {
                    "torch": torch.get_rng_state(),
                    "cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
                    "numpy": (np.random.get_state() if "numpy.random" in sys.modules else None),
                },
            }

            # Save the best checkpoint
            best_checkpoint_path = checkpoint_dir / "best_checkpoint.pt"
            torch.save(best_checkpoint, best_checkpoint_path)

            # If optimizer state saving is enabled, save the optimizer state to the optimizer directory
            if args.save_optimizer_state:
                best_optimizer_state = {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "steps": steps,
                    "epoch": epoch,
                    "best_loss": best_loss,
                }
                best_optimizer_state_path = optimizer_dir / "best_optimizer_state.pt"
                torch.save(best_optimizer_state, best_optimizer_state_path)

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

        # Log to wandb if enabled
        if args.wandb:
            log_to_wandb(logging_dict, steps, "valid")

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
    with safe_globals([Namespace, np.ndarray, np.dtype, np._core.multiarray._reconstruct]):
        best_checkpoint_path = checkpoint_dir / "best_checkpoint.pt"
        dicts = torch.load(best_checkpoint_path, weights_only=False)

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
            with _get_autocast_context(device):
                outs = test_model(**device_batch)

            # Calculate loss here (outside autocast for precision)
            loss = F.nll_loss(
                F.log_softmax(outs.view(-1, outs.size(-1)), dim=-1, dtype=torch.float32),
                targets.view(-1),
                reduction="sum",
                ignore_index=tokenizer.pad_token_id,
            )

            pred_tokens = _count_pred_tokens(targets, tokenizer.pad_token_id)
            if pred_tokens > 0:
                normal_loss = loss.item() / pred_tokens / math.log(2)
                normal_acc = (
                    accuracy(outs, targets, ignore_index=tokenizer.pad_token_id) / pred_tokens
                )
                # Update the meters appropriately
                meters["test_loss"].update(normal_loss, pred_tokens)
                meters["test_acc"].update(normal_acc, pred_tokens)

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

    # Log to wandb if enabled
    if args.wandb:
        log_to_wandb(logging_dict, steps, "test")

    LOGGER.info(
        f"Training is finished! See output in {args.checkpoint_dir} and "
        f"tensorboard logs in {args.tensorboard_log_dir}"
    )

    # Finish wandb run if active
    if args.wandb and wandb.run is not None:
        wandb.finish()


def cli_main() -> None:
    """CLI entrypoint for MPNet pretraining.

    :return None: This function returns nothing.
    """
    parser = argparse.ArgumentParser(
        description="Pretrain an MPNet model with a huggingface dataset "
        "or path(s) to local training/eval data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Default args follow the MPNet-base params described in the paper: "
        "https://arxiv.org/abs/2004.09297",
    )
    parser.add_argument(
        "--encoder-layers",
        help="The number of encoder layers within the encoder block of MPNet. Subsequent "
        "papers typically use 12-24 encoder layers.",
        default=12,
        type=int,
    )
    parser.add_argument(
        "--encoder-embed-dim",
        help="The dimension of the embedding layer inside each encoder block. Generally 768-1024.",
        default=768,
        type=int,
    )
    parser.add_argument(
        "--encoder-ffn-dim",
        help="The dimension of the feed-forward hidden layer after each self-attention "
        "calculation. Typically 3-4x the embedding dimension",
        default=3072,
        type=int,
    )
    parser.add_argument(
        "--encoder-attention-heads",
        help="The number of attention heads in each layer. Typically 8-16",
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
        "--tokenizer-name",
        help="The name of the tokenizer to use. This should be a HuggingFace tokenizer name, "
        "e.g. 'microsoft/mpnet-base' or a path to a local tokenizer/model directory.",
        default="microsoft/mpnet-base",
        type=str,
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
        "-num_buckets",
        "--relative-attention-num-buckets",
        help="Number of buckets for relative position. If not set, will automatically compute "
        "the number of buckets based on the max sequence length.",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--relative-attention-max-distance",
        help="Maximum distance for relative position. If not set, will automatically compute "
        "the maximum distance based on the max sequence length.",
        default=None,
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
        default=min(4, os.cpu_count()),
        type=int,
    )
    parser.add_argument(
        "--compile",
        help="Whether or not to compile the model",
        action="store_true",
        default=False,
    )

    # Resumable training arguments
    parser.add_argument(
        "--resume",
        help="Whether to resume training from a checkpoint",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--resume-checkpoint",
        help="Path to the checkpoint to resume from. If not provided, will use best_checkpoint.pt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--hf-model-path",
        help="Path to a HuggingFace MPNet model to initialize weights from (alternative to resuming from repo checkpoint)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save-optimizer-state",
        help="Whether to save optimizer state for resumable training",
        action="store_true",
        default=False,
    )

    # Weights & Biases arguments
    parser.add_argument(
        "--wandb",
        help="Whether to use Weights & Biases for logging",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--wandb-project",
        help="Weights & Biases project name",
        default="annotated-mpnet",
        type=str,
    )
    parser.add_argument(
        "--wandb-name",
        help="Weights & Biases run name",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--wandb-id",
        help="Weights & Biases run ID for resuming a run",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--wandb-watch",
        help="Whether to log model gradients in Weights & Biases",
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

    # Check for validity of resumable training arguments
    if args.resume and args.hf_model_path:
        LOGGER.warning(
            "Both --resume and --hf-model-path are specified. "
            "HuggingFace model will take precedence over resuming from checkpoint."
        )

    if args.resume_checkpoint and not args.resume:
        LOGGER.warning(
            "--resume-checkpoint provided but --resume flag not set. "
            "Will not resume from checkpoint. Did you mean to add --resume?"
        )

    LOGGER.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()
