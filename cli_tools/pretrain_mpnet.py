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
import random
import sys
from argparse import Namespace
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
from rich.logging import RichHandler

LOG_FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=LOG_FORMAT, datefmt="[%X] ", handlers=[RichHandler()])
LOGGER = logging.getLogger(__name__)

DEFAULT_STREAMING_DATASET = "HuggingFaceFW/fineweb-edu"


import torch
import torch.nn.functional as F
from datasets import load_dataset
from rich.progress import track
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
    hf_max_positions_to_internal,
    model_summary,
    validate_tokenizer,
)

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


DEFAULT_BEST_LOSS = 10e6
VOCAB_SIZE_ALIGNMENT = 128  # Align vocab size for efficient GPU kernels.

# Optional dependencies are imported lazily; keep a module-level slot for wandb.
wandb = None


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


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs.

    :param int seed: Seed value to apply.
    :return None: This function returns nothing.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(payload: Any, path: pathlib.Path) -> None:
    """Write a torch checkpoint atomically to avoid partial files.

    :param Any payload: Object to serialize.
    :param pathlib.Path path: Destination path.
    :return None: This function returns nothing.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def write_to_tensorboard(writer: "SummaryWriter", logging_dict: dict, step: int) -> None:
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
    if wandb is not None and wandb.run is not None:
        # Prefix metrics with split name for better organization in the dashboard
        wandb_dict = {f"{split}/{k}": v for k, v in logging_dict.items()}
        wandb_dict["step"] = step
        wandb.log(wandb_dict)


def _group_parameters_for_weight_decay(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split parameters into decay and no-decay groups.

    :param torch.nn.Module model: Model whose parameters should be grouped.
    :return tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]: (decay, no_decay) params.
    """
    decay_params = []
    no_decay_params = []
    seen_params = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_id = id(param)
        if param_id in seen_params:
            continue
        seen_params.add(param_id)
        if name.endswith(".bias") or param.ndim == 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return decay_params, no_decay_params


def _get_initial_best_loss(checkpoint: dict | None) -> float:
    """Return the best loss from a checkpoint or a default value.

    :param dict checkpoint: Loaded checkpoint or None.
    :return float: Best loss value.
    """
    if checkpoint is None:
        return DEFAULT_BEST_LOSS

    return checkpoint.get("best_loss", DEFAULT_BEST_LOSS)


def _strip_compile_prefix(model_states: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip torch.compile prefixes from state dict keys.

    :param dict model_states: Raw model state dict.
    :return dict: State dict with compile prefixes removed.
    """
    return {k.replace("_orig_mod.", ""): v for k, v in model_states.items()}


def _coerce_rng_state(rng_state: Any) -> torch.ByteTensor:
    """Coerce RNG state into a CPU uint8 tensor for torch.set_rng_state.

    :param Any rng_state: RNG state payload.
    :return torch.ByteTensor: CPU uint8 RNG tensor.
    """
    if isinstance(rng_state, torch.Tensor):
        rng_state = rng_state.detach().cpu()
    return torch.as_tensor(rng_state, dtype=torch.uint8)


def _serialize_numpy_rng_state(rng_state: Any) -> Any:
    """Serialize numpy RNG state into builtin types for safe checkpoint loading.

    :param Any rng_state: Numpy RNG state payload.
    :return Any: Serialized RNG state payload.
    """
    if rng_state is None:
        return None
    if isinstance(rng_state, tuple) and len(rng_state) == 5:
        algo, state, pos, has_gauss, cached_gaussian = rng_state
        if isinstance(state, np.ndarray):
            return {
                "algorithm": algo,
                "state": state.tolist(),
                "pos": int(pos),
                "has_gauss": int(has_gauss),
                "cached_gaussian": float(cached_gaussian),
            }
    return rng_state


def _deserialize_numpy_rng_state(rng_state: Any) -> Any:
    """Deserialize numpy RNG state from builtin types.

    :param Any rng_state: Serialized RNG state payload.
    :return Any: Numpy RNG state tuple or original payload.
    """
    if rng_state is None:
        return None
    if isinstance(rng_state, dict) and "state" in rng_state:
        return (
            rng_state["algorithm"],
            np.array(rng_state["state"], dtype=np.uint32),
            rng_state["pos"],
            rng_state["has_gauss"],
            rng_state["cached_gaussian"],
        )
    return rng_state


def _safe_torch_load(
    path: pathlib.Path, map_location: str | torch.device, trust_checkpoint: bool
) -> dict:
    """Load a checkpoint with weights_only by default, optionally allowing unsafe fallback.

    :param pathlib.Path path: Path to the checkpoint file.
    :param str | torch.device map_location: Map location for loading tensors.
    :param bool trust_checkpoint: Whether to allow unsafe loading fallback.
    :return dict: Loaded checkpoint payload.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        if trust_checkpoint:
            LOGGER.warning(
                "Falling back to unsafe torch.load for %s because --trust-checkpoint was set: %s",
                path,
                exc,
            )
            return torch.load(path, map_location=map_location, weights_only=False)
        raise RuntimeError(
            f"Failed to load checkpoint {path} with safe weights_only loading. "
            "Re-export the checkpoint with this version or pass --trust-checkpoint to allow "
            "unsafe loading."
        ) from exc


def _resolve_best_loss(
    checkpoint: dict | None,
    checkpoint_dir: pathlib.Path,
    resume_checkpoint_path: pathlib.Path | None = None,
    trust_checkpoint: bool = False,
) -> float:
    """Resolve the best loss from a checkpoint or the best checkpoint file.

    :param dict checkpoint: Loaded checkpoint or None.
    :param pathlib.Path checkpoint_dir: Directory containing checkpoints.
    :param pathlib.Path resume_checkpoint_path: Resume checkpoint path, defaults to None.
    :param bool trust_checkpoint: Whether to allow unsafe checkpoint loading, defaults to False.
    :return float: Best loss value.
    """
    best_checkpoint_root = checkpoint_dir
    if resume_checkpoint_path is not None:
        resume_root = resume_checkpoint_path.parent
        if resume_root.resolve() != checkpoint_dir.resolve():
            best_checkpoint_root = resume_root

    best_checkpoint_path = best_checkpoint_root / "best_checkpoint.pt"
    if best_checkpoint_path.exists():
        try:
            best_checkpoint = _safe_torch_load(
                best_checkpoint_path, map_location="cpu", trust_checkpoint=trust_checkpoint
            )
            return _get_initial_best_loss(best_checkpoint)
        except (OSError, RuntimeError, ValueError) as exc:
            LOGGER.warning(f"Could not load best checkpoint for best_loss: {exc}")

    return _get_initial_best_loss(checkpoint)


def _normalize_data_state(
    data_state: dict | None, mode_hint: str | None = None
) -> dict[str, int | str]:
    """Normalize data_state values for resume logic.

    :param dict data_state: Raw data_state dictionary.
    :param str mode_hint: Optional mode hint ("streaming" or "files"), defaults to None.
    :return dict[str, int | str]: Normalized data_state.
    """
    data_state = data_state if isinstance(data_state, dict) else {}
    return {
        "mode": data_state.get("mode", mode_hint or "unknown"),
        "cycle": int(data_state.get("cycle", 0) or 0),
        "batch_index": int(data_state.get("batch_index", 0) or 0),
        "samples_in_cycle": int(data_state.get("samples_in_cycle", 0) or 0),
        "legacy": bool(data_state.get("legacy", False)),
    }


def _get_resume_metadata(
    checkpoint: dict, resume_checkpoint_path: pathlib.Path | None
) -> tuple[int, dict[str, int | str]]:
    """Return resume metadata with legacy checkpoint fallback.

    :param dict checkpoint: Loaded checkpoint data.
    :param pathlib.Path resume_checkpoint_path: Checkpoint path, defaults to None.
    :return tuple[int, dict[str, int | str]]: samples_processed and normalized data_state.
    """
    data_state = checkpoint.get("data_state")
    if isinstance(data_state, dict):
        normalized = _normalize_data_state(data_state)
    else:
        missing_fields = [
            field
            for field in ("samples_processed", "epoch_batches_processed", "epoch_complete")
            if field not in checkpoint
        ]
        if missing_fields:
            label = (
                str(resume_checkpoint_path) if resume_checkpoint_path is not None else "checkpoint"
            )
            LOGGER.warning(
                "Legacy checkpoint format detected for %s (missing %s). "
                "Legacy resume is no longer supported; this checkpoint can only be used to "
                "initialize weights.",
                label,
                ", ".join(missing_fields),
            )
        legacy_cycle = int(checkpoint.get("epoch", 0) or 0)
        legacy_batches = int(checkpoint.get("epoch_batches_processed", 0) or 0)
        legacy_complete = bool(checkpoint.get("epoch_complete", False))
        if legacy_complete:
            legacy_cycle += 1
            legacy_batches = 0
        normalized = _normalize_data_state(
            {
                "cycle": legacy_cycle,
                "batch_index": legacy_batches,
                "samples_in_cycle": 0,
                "mode": "legacy",
                "legacy": True,
            }
        )
        if legacy_batches:
            LOGGER.warning(
                "Legacy resume metadata does not include per-cycle sample counts; "
                "streaming resumes are not supported and will be reinitialized."
            )

    return int(checkpoint.get("samples_processed", 0) or 0), normalized


def _should_save_checkpoint(steps: int, checkpoint_interval: int) -> bool:
    """Return whether a checkpoint should be saved at the current step.

    :param int steps: Number of completed update steps.
    :param int checkpoint_interval: Interval for checkpointing.
    :return bool: True if a checkpoint should be written.
    """
    return checkpoint_interval > 0 and steps > 0 and steps % checkpoint_interval == 0


def _checkpoint_step_from_path(path: pathlib.Path) -> int:
    """Extract the step number from a checkpoint filename.

    :param pathlib.Path path: Checkpoint path.
    :return int: Parsed step number or -1 if not parseable.
    """
    stem = path.stem
    if stem.startswith("checkpoint"):
        step_str = stem[len("checkpoint") :]
        if step_str.isdigit():
            return int(step_str)
    return -1


def _find_latest_checkpoint(checkpoint_dir: pathlib.Path) -> pathlib.Path | None:
    """Return the latest interval checkpoint in a directory.

    :param pathlib.Path checkpoint_dir: Directory containing checkpoint files.
    :return pathlib.Path | None: Latest checkpoint path or None if not found.
    """
    checkpoints = [
        checkpoint
        for checkpoint in checkpoint_dir.glob("checkpoint*.pt")
        if _checkpoint_step_from_path(checkpoint) >= 0
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=_checkpoint_step_from_path)


def _prune_checkpoints(
    checkpoint_dir: pathlib.Path,
    keep_checkpoints: int,
    optimizer_dir: pathlib.Path | None = None,
) -> None:
    """Delete older interval checkpoints, keeping only the most recent N.

    :param pathlib.Path checkpoint_dir: Directory containing checkpoint files.
    :param int keep_checkpoints: Number of recent checkpoints to keep (-1 disables pruning).
    :param pathlib.Path optimizer_dir: Optimizer state directory to prune alongside checkpoints.
    :return None: This function returns nothing.
    """
    if keep_checkpoints < 0:
        return

    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint*.pt"),
        key=_checkpoint_step_from_path,
    )
    if keep_checkpoints == 0:
        to_remove = checkpoints
    else:
        to_remove = checkpoints[:-keep_checkpoints]

    for ckpt in to_remove:
        step = _checkpoint_step_from_path(ckpt)
        try:
            ckpt.unlink()
        except FileNotFoundError:
            continue
        if optimizer_dir is not None and step >= 0:
            optimizer_state = optimizer_dir / f"checkpoint{step}_optimizer_state.pt"
            if optimizer_state.exists():
                optimizer_state.unlink()


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
    args.relative_attention_max_distance = checkpoint_args.get(
        "relative_attention_max_distance",
        getattr(args, "relative_attention_max_distance", None),
    )
    args.normalize_before = checkpoint_args.get(
        "normalize_before", getattr(args, "normalize_before", False)
    )
    args.original_vocab_size = checkpoint_args.get("original_vocab_size", args.original_vocab_size)
    args.padded_vocab_size = checkpoint_args.get("padded_vocab_size", args.padded_vocab_size)

    args.max_tokens = checkpoint_args.get("max_tokens", args.max_tokens)
    if "max_positions" in checkpoint_args:
        args.max_positions = checkpoint_args.get("max_positions", args.max_positions)
    else:
        args.max_positions = args.max_tokens


def _validate_tokenizer_vocab_size(tokenizer: Any, args: Namespace, source: str) -> None:
    """Validate tokenizer vocabulary size matches model checkpoint/config.

    :param Any tokenizer: Tokenizer instance used for training.
    :param Namespace args: Parsed CLI args with vocab sizing.
    :param str source: Source label for error messaging.
    :raises ValueError: If tokenizer vocab size does not match the checkpoint/config size.
    """
    tokenizer_vocab_size = len(tokenizer)
    expected_vocab_size = args.original_vocab_size
    if tokenizer_vocab_size != expected_vocab_size:
        raise ValueError(
            f"Tokenizer vocab size ({tokenizer_vocab_size}) does not match {source} vocab size "
            f"({expected_vocab_size}). Use the same tokenizer as the {source} or regenerate the "
            "checkpoint/config."
        )


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
    :raises FileNotFoundError: If no resume checkpoint can be found.
    :raises IsADirectoryError: If resume checkpoint points to a directory.
    """
    if resume_checkpoint is None:
        best_checkpoint_path = checkpoint_dir / "best_checkpoint.pt"
        if best_checkpoint_path.exists():
            return best_checkpoint_path
        latest_checkpoint = _find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint is not None:
            LOGGER.warning(
                "Best checkpoint not found at %s; falling back to latest interval checkpoint %s.",
                best_checkpoint_path,
                latest_checkpoint,
            )
            return latest_checkpoint
        raise FileNotFoundError(
            f"No resume checkpoint found. Expected {best_checkpoint_path} or interval checkpoints "
            f"in {checkpoint_dir}."
        )
    resume_checkpoint_path = pathlib.Path(resume_checkpoint)
    if resume_checkpoint_path.is_dir():
        raise IsADirectoryError(
            f"Resume checkpoint path {resume_checkpoint_path} is a directory; expected a .pt file."
        )
    if not resume_checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint {resume_checkpoint_path} not found.")
    return resume_checkpoint_path


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


def _resolve_optimizer_state_dir(
    checkpoint_dir: pathlib.Path, resume_checkpoint_path: pathlib.Path
) -> pathlib.Path:
    """Resolve optimizer state directory for a resume checkpoint.

    :param pathlib.Path checkpoint_dir: Current output checkpoint directory.
    :param pathlib.Path resume_checkpoint_path: Checkpoint being resumed.
    :return pathlib.Path: Optimizer state directory to use.
    """
    if resume_checkpoint_path.parent.resolve() != checkpoint_dir.resolve():
        return resume_checkpoint_path.parent / "optimizer"
    return checkpoint_dir / "optimizer"


def _get_optimizer_state_path_for_resume(
    checkpoint_dir: pathlib.Path, resume_checkpoint_path: pathlib.Path
) -> pathlib.Path:
    """Return optimizer state path for a resume checkpoint.

    :param pathlib.Path checkpoint_dir: Current output checkpoint directory.
    :param pathlib.Path resume_checkpoint_path: Checkpoint being resumed.
    :return pathlib.Path: Optimizer state path to load if available.
    """
    optimizer_state_dir = _resolve_optimizer_state_dir(checkpoint_dir, resume_checkpoint_path)
    return _select_optimizer_state_path(optimizer_state_dir, resume_checkpoint_path)


def _select_best_checkpoint_path(
    checkpoint_dir: pathlib.Path, resume_checkpoint_path: pathlib.Path | None
) -> pathlib.Path:
    """Select a best checkpoint path, falling back to resume root if needed.

    :param pathlib.Path checkpoint_dir: Current output checkpoint directory.
    :param pathlib.Path resume_checkpoint_path: Resume checkpoint path, defaults to None.
    :return pathlib.Path: Best checkpoint path to load for final eval.
    """
    best_checkpoint_path = checkpoint_dir / "best_checkpoint.pt"
    if best_checkpoint_path.exists():
        return best_checkpoint_path

    if resume_checkpoint_path is not None:
        resume_root = resume_checkpoint_path.parent
        resume_best = resume_root / "best_checkpoint.pt"
        if resume_best.exists():
            return resume_best

    return best_checkpoint_path


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
    # BF16 is the default mixed-precision path on CUDA; GradScaler is unnecessary for BF16.
    # If you switch to FP16, add a GradScaler and unscale before clipping.
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _ensure_bf16_supported(device: torch.device) -> None:
    """Fail fast if BF16 is unavailable on the selected CUDA device.

    :param torch.device device: Device used for training/inference.
    :return None: This function returns nothing.
    :raises RuntimeError: If BF16 is not supported on the active CUDA device.
    """
    if device.type != "cuda":
        return
    if torch.cuda.is_bf16_supported():
        return
    try:
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)
        major, minor = torch.cuda.get_device_capability(device_index)
        details = f"{device_name} (compute capability {major}.{minor})"
    except Exception:
        details = "the selected CUDA device"
    raise RuntimeError(
        "BF16 is required for MPNet pretraining in this repo (legacy GPUs are not supported "
        f"as of 2026). Detected {details} without BF16 support. Use a BF16-capable GPU (Ampere+)."
    )


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

    _ensure_bf16_supported(device)
    check_and_activate_tf32()  # Check if the GPU supports NVIDIA Ampere or later and enable TF32

    # Seed all RNGs early for reproducible initialization and data ordering.
    _seed_everything(args.seed)

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
    assert is_valid, (
        f"Tokenizer validation failed for {args.tokenizer_name}. "
        "Run validate_tokenizer(tokenizer, verbose=True) for details."
    )
    if not details.get("whole_word_mask", False):
        # Tokenizer can be valid but incompatible with WordPiece-style whole-word masking.
        LOGGER.warning(
            "Tokenizer %s does not appear to support whole-word masking; "
            "the data collator will fall back to token-level masking.",
            args.tokenizer_name,
        )

    # Get the tokenizer vocab size
    original_vocab_size = len(
        tokenizer
    )  # Use len() to get actual vocab size including added tokens

    # Determine whether to pad vocab size based on whether we're loading from existing model
    if args.resume or args.hf_model_path is not None:
        # When loading from existing model, preserve its vocab size to avoid mismatched embeddings.
        LOGGER.info("Loading from existing model - will use model's vocab size")
        # These will be overridden when we load the checkpoint or HF config
        args.original_vocab_size = original_vocab_size
        args.padded_vocab_size = original_vocab_size
    else:
        # When training from scratch, pad vocab size for GPU performance
        target_vocab_size = (
            (original_vocab_size + VOCAB_SIZE_ALIGNMENT - 1) // VOCAB_SIZE_ALIGNMENT
        ) * VOCAB_SIZE_ALIGNMENT

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
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError(
                "Tensorboard logging requested but tensorboard dependencies are missing. "
                "Install torch[torchvision] extras or tensorboard to enable this feature."
            ) from exc

        log_dir = pathlib.Path(args.tensorboard_log_dir)
        writers = {
            "train": SummaryWriter(str(log_dir / "train")),
            "valid": SummaryWriter(str(log_dir / "valid")),
            "test": SummaryWriter(str(log_dir / "test")),
        }

    # Check if we're resuming and need to load architecture from checkpoint
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    arch_source = _select_architecture_source(args)
    resume_checkpoint_path: pathlib.Path | None = None
    resume_checkpoint: dict | None = None
    resume_samples_processed = 0
    resume_data_state: dict[str, int | str] = _normalize_data_state(None)

    if arch_source == "resume":
        # Load checkpoint to get architecture before creating model
        resume_checkpoint_path = _select_resume_checkpoint_path(
            checkpoint_dir, args.resume_checkpoint
        )

        LOGGER.info(f"Loading architecture from checkpoint: {resume_checkpoint_path}")
        resume_checkpoint = _safe_torch_load(
            resume_checkpoint_path,
            map_location="cpu",
            trust_checkpoint=args.trust_checkpoint,
        )

        resume_samples_processed, resume_data_state = _get_resume_metadata(
            resume_checkpoint, resume_checkpoint_path
        )

        # Restore architecture args from checkpoint
        if "args" in resume_checkpoint:
            checkpoint_args = resume_checkpoint["args"]
            checkpoint_args_dict = (
                vars(checkpoint_args) if isinstance(checkpoint_args, Namespace) else checkpoint_args
            )
            _apply_checkpoint_architecture_args(args, checkpoint_args)
            # Update tokenizer length immediately after restoring checkpoint args.
            tokenizer.model_max_length = args.max_tokens
            _warn_if_max_positions_mismatch(args)
            _validate_tokenizer_vocab_size(tokenizer, args, "checkpoint")
            checkpoint_tokenizer_name = checkpoint_args_dict.get("tokenizer_name")
            if (
                checkpoint_tokenizer_name is not None
                and checkpoint_tokenizer_name != args.tokenizer_name
            ):
                LOGGER.warning(
                    "Checkpoint tokenizer (%s) does not match current --tokenizer-name (%s). "
                    "If vocab sizes differ, resume will fail.",
                    checkpoint_tokenizer_name,
                    args.tokenizer_name,
                )

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
        # HF config includes special tokens; internal max_positions excludes them.
        args.max_positions = hf_max_positions_to_internal(hf_config.max_position_embeddings)
        if args.max_tokens > args.max_positions:
            LOGGER.warning(
                "max_tokens exceeds HuggingFace max_position_embeddings; clamping max_tokens to "
                f"{args.max_positions} to avoid position embedding mismatches."
            )
            args.max_tokens = args.max_positions
            tokenizer.model_max_length = args.max_tokens
        args.relative_attention_num_buckets = getattr(
            hf_config, "relative_attention_num_buckets", 32
        )
        args.original_vocab_size = hf_config.vocab_size
        args.padded_vocab_size = hf_config.vocab_size
        _validate_tokenizer_vocab_size(tokenizer, args, "HuggingFace model")

        LOGGER.info(
            f"Using HF model architecture: {args.encoder_layers} layers, "
            f"{args.encoder_embed_dim} hidden, {args.encoder_ffn_dim} FFN"
        )

    # Next, we instantiate the model and the data collator
    model = MPNetForPretraining(args, tokenizer)
    train_collator = DataCollatorForMaskedPermutedLanguageModeling(
        tokenizer=tokenizer, random_seed=args.seed
    )
    eval_collator = DataCollatorForMaskedPermutedLanguageModeling(
        tokenizer=tokenizer, random_seed=args.seed + 1
    )
    eval_collator_state = eval_collator.get_rng_state()

    # Initialize wandb if enabled (after model creation)
    if args.wandb:
        global wandb
        try:
            import wandb as _wandb
        except ImportError as exc:
            raise RuntimeError(
                "Weights & Biases logging requested but wandb is not installed. "
                "Install wandb to enable this feature."
            ) from exc
        wandb = _wandb
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

    # Determine whether to use streaming dataset or file-based dataset
    if args.dataset_name:
        LOGGER.info(f"Using HuggingFace dataset: {args.dataset_name}")

        try:
            # Load the dataset ONCE in streaming mode (streaming datasets are lazy and cheap to keep).
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
                collate_fn=eval_collator,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                collate_fn=eval_collator,
            )

            # Skip validation/test samples from the raw stream so we never train on them.
            stream_skip_samples = args.eval_samples * 2
            if stream_skip_samples > 0:
                train_stream = train_stream.skip(stream_skip_samples)
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
            valid_dataset, collate_fn=eval_collator, batch_size=args.batch_size
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, collate_fn=eval_collator, batch_size=args.batch_size
        )

        # Get each of the files in the training directory
        train_dir = pathlib.Path(args.train_dir)
        train_files = sorted(str(path) for path in train_dir.iterdir() if path.is_file())

    has_validation = len(valid_dataloader) > 0
    has_test = len(test_dataloader) > 0
    if not has_validation:
        LOGGER.warning(
            "Validation dataloader is empty; skipping validation and best-checkpoint tracking."
        )
        if args.checkpoint_interval <= 0:
            LOGGER.warning(
                "No interval checkpoints will be written with --checkpoint-interval <= 0; "
                "resume checkpoints will not be available."
            )
        elif args.keep_checkpoints == 0:
            LOGGER.warning(
                "Interval checkpoints will be pruned immediately with --keep-checkpoints=0; "
                "no resume checkpoints will remain."
            )
    if not has_test:
        LOGGER.warning("Test dataloader is empty; skipping final test evaluation.")

    # Note: checkpoint_dir is already created above when handling resume logic

    resume_mode = "streaming" if train_streaming else "files"
    # Exact resume is only possible with deterministic sampling (no streaming shuffle, num_workers=0).
    if args.resume and train_streaming:
        LOGGER.warning("Streaming shuffle buffers are not restorable; resume will be approximate.")
    if args.resume and args.num_workers > 0:
        LOGGER.warning(
            "Collator RNG state is not restorable with num_workers=%s; "
            "use --num-workers 0 for deterministic resume.",
            args.num_workers,
        )
    resume_data_state = _normalize_data_state(resume_data_state, mode_hint=resume_mode)
    reset_collator_rng = False  # Flag to skip loading RNG states from checkpoint
    if resume_data_state["mode"] == "unknown":
        resume_data_state["mode"] = resume_mode
    elif resume_data_state["mode"] != resume_mode:
        LOGGER.warning(
            "Resume checkpoint data_state mode (%s) does not match current dataset (%s). "
            "Resetting resume offsets and RNG states to ensure consistency.",
            resume_data_state["mode"],
            resume_mode,
        )
        resume_data_state = _normalize_data_state(
            {
                "mode": resume_mode,
                "cycle": 0,
                "batch_index": 0,
                "samples_in_cycle": 0,
                "legacy": False,
            },
            mode_hint=resume_mode,
        )

        # Reset all RNG to fresh seed for consistency after mode mismatch.
        # This ensures data sampling is reproducible from a clean state.
        base_seed = args.seed
        torch.manual_seed(base_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(base_seed)
        np.random.seed(base_seed)
        LOGGER.info("Reset all RNG states due to mode mismatch")

        # Set flag to skip loading RNG states from checkpoint.
        reset_collator_rng = True
    resume_cycle_batch_index = int(resume_data_state.get("batch_index", 0) or 0)
    resume_cycle_samples = int(resume_data_state.get("samples_in_cycle", 0) or 0)
    # Legacy checkpoints (no data_state) are not resumable; we only use them to initialize weights.
    legacy_resume = bool(resume_data_state.get("legacy", False))

    # Create optimizer state directory if saving optimizer states
    if args.save_optimizer_state:
        optimizer_dir = checkpoint_dir / "optimizer"
        optimizer_dir.mkdir(parents=True, exist_ok=True)

    # Before defining the scheduler and optimizer, let's make sure warmup_updates is set. If it
    # isn't, we need to set it to 10% the amount of total_updates
    if args.warmup_updates is None:
        args.warmup_updates = round(0.1 * args.total_updates)

    # Let's define an optimizer with our Polynomial decay scheduler on top of it.
    # We set the optimizer with an arbitrary learning rate since it will be updated by the scheduler.
    # Use parameter groups to avoid weight decay on biases and norm weights.
    decay_params = []
    no_decay_params = []
    seen_params = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_id = id(param)
        if param_id in seen_params:
            continue
        seen_params.add(param_id)
        if name.endswith(".bias") or param.ndim == 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        betas=(args.beta1, args.beta2),
        lr=6e-9,  # starting learning rate during warmup
        eps=args.adam_eps,
        fused=device.type == "cuda",
    )
    scheduler = PolynomialDecayLRScheduler(args, optimizer)

    # Initialize step counters (completed optimizer updates).
    steps = 0
    best_loss = DEFAULT_BEST_LOSS  # Will be overridden if resuming from checkpoint
    samples_processed = 0

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

            checkpoint = _safe_torch_load(
                temp_checkpoint_path,
                map_location=device,
                trust_checkpoint=args.trust_checkpoint,
            )

            # Extract model states
            model_states = _strip_compile_prefix(checkpoint["model_states"])

            # Load model weights; strict loading will surface any key mismatches.
            model.load_state_dict(model_states)
            LOGGER.info("Model weights loaded successfully from HuggingFace model")

        except Exception as e:
            LOGGER.error(f"Error loading HuggingFace model: {e}")
            LOGGER.error(
                "Failed to initialize from HuggingFace model. Remove --hf-model-path to use "
                "random initialization."
            )
            raise

    # Handle resuming from checkpoint if enabled
    elif args.resume:
        # Note: We already loaded the checkpoint above to get architecture
        # Now we just need to extract the other state
        checkpoint = resume_checkpoint
        if checkpoint is None:
            raise RuntimeError("Resume requested but no checkpoint was loaded.")

        if legacy_resume:
            LOGGER.warning(
                "Resume requested for a legacy checkpoint (no data_state). "
                "Full resume is not supported; loading weights only and reinitializing "
                "optimizer/scheduler/step counters."
            )
            steps = 0
            samples_processed = 0
            best_loss = DEFAULT_BEST_LOSS
            resume_data_state = _normalize_data_state(
                {
                    "mode": resume_mode,
                    "cycle": 0,
                    "batch_index": 0,
                    "samples_in_cycle": 0,
                    "legacy": False,
                },
                mode_hint=resume_mode,
            )
            resume_cycle_batch_index = 0
            resume_cycle_samples = 0
        else:
            # Extract training state
            if "steps" in checkpoint:
                steps = checkpoint["steps"]
                LOGGER.info(f"Resuming from step {steps}")

            samples_processed = resume_samples_processed

            best_loss = _resolve_best_loss(
                checkpoint,
                checkpoint_dir,
                resume_checkpoint_path,
                trust_checkpoint=args.trust_checkpoint,
            )
            if best_loss != DEFAULT_BEST_LOSS:
                LOGGER.info(f"Best validation loss from checkpoint: {best_loss}")

            # Restore RNG state if available
            if "rng_state" in checkpoint:
                # RNG restoration is best-effort; warn and continue to avoid aborting training.
                try:
                    rng_state = checkpoint["rng_state"]
                    if rng_state.get("torch") is not None:
                        torch_rng = rng_state["torch"]
                        torch.set_rng_state(_coerce_rng_state(torch_rng))
                        LOGGER.info("Restored torch RNG state")

                    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
                        cuda_states = rng_state["cuda"]
                        # Handle list of CUDA states for multi-GPU
                        if isinstance(cuda_states, list):
                            processed_states = []
                            for state in cuda_states:
                                processed_states.append(_coerce_rng_state(state))
                            torch.cuda.set_rng_state_all(processed_states)
                        else:
                            torch.cuda.set_rng_state(_coerce_rng_state(cuda_states))
                        LOGGER.info("Restored CUDA RNG state")

                    if "numpy.random" in sys.modules and rng_state.get("numpy") is not None:
                        np.random.set_state(_deserialize_numpy_rng_state(rng_state["numpy"]))
                        LOGGER.info("Restored numpy RNG state")
                except (TypeError, ValueError, AttributeError) as e:
                    LOGGER.warning(f"Could not restore RNG state: {e}")
                    LOGGER.info("Continuing with current RNG state")

            # Restore collator RNG state for deterministic masking when resuming.
            # Skip if mode mismatch was detected (reset_collator_rng is True).
            if "collator_rng_state" in checkpoint and not reset_collator_rng:
                if args.num_workers > 0:
                    LOGGER.info(
                        "Skipping collator RNG restoration with num_workers=%s; "
                        "worker RNG streams cannot be restored from the main process.",
                        args.num_workers,
                    )
                else:
                    try:
                        train_collator.set_rng_state(checkpoint["collator_rng_state"])
                        LOGGER.info("Restored collator RNG state")
                    except (TypeError, ValueError, AttributeError) as e:
                        LOGGER.warning(f"Could not restore collator RNG state: {e}")
            elif reset_collator_rng:
                LOGGER.info("Skipped collator RNG restoration due to mode mismatch")

        # Extract model states
        model_states = _strip_compile_prefix(checkpoint["model_states"])

        # Load model weights; strict loading will surface any key mismatches.
        model.load_state_dict(model_states)
        LOGGER.info("Model weights loaded successfully")

        if not legacy_resume:
            # Load optimizer state if present; save flag only controls writing new state files.
            optimizer_state_dir = _resolve_optimizer_state_dir(
                checkpoint_dir, resume_checkpoint_path
            )
            expected_optimizer_dir = checkpoint_dir / "optimizer"
            if optimizer_state_dir.resolve() != expected_optimizer_dir.resolve():
                LOGGER.warning(
                    "Resume checkpoint is outside checkpoint_dir; looking for optimizer state in "
                    f"{optimizer_state_dir}."
                )
            optimizer_state_path = _select_optimizer_state_path(
                optimizer_state_dir, resume_checkpoint_path
            )
            if optimizer_state_path.exists():
                LOGGER.info(f"Loading optimizer state from {optimizer_state_path}")
                optimizer_state = _safe_torch_load(
                    optimizer_state_path,
                    map_location=device,
                    trust_checkpoint=args.trust_checkpoint,
                )

                # Load optimizer state
                optimizer.load_state_dict(optimizer_state["optimizer"])

                # Load scheduler state
                # Scheduler is stateless; load_state_dict is kept for legacy state dicts/overrides.
                scheduler.load_state_dict(optimizer_state["scheduler"])

                LOGGER.info("Optimizer and scheduler states loaded successfully")
            else:
                LOGGER.warning(
                    f"No optimizer state found at {optimizer_state_path}, using default initialization"
                )

    # Compile after any checkpoint/HF weight loading so state dict keys stay consistent.
    if args.compile:
        LOGGER.info("Compiling the model...")
        model = torch.compile(model)

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

    def _select_model_inputs(batch: dict[str, Any]) -> dict[str, Any]:
        """Select only the inputs required by the pretraining model.

        :param dict[str, Any] batch: Full batch dictionary from the collator.
        :return dict[str, Any]: Filtered model inputs.
        """
        model_inputs = {
            "input_ids": batch["input_ids"],
            "positions": batch["positions"],
            "pred_size": batch["pred_size"],
        }
        if "attention_mask" in batch:
            model_inputs["attention_mask"] = batch["attention_mask"]
        return model_inputs

    def _evaluate_split(
        model_to_eval: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        split_name: str,
        loss_key: str,
        acc_key: str,
        collator: DataCollatorForMaskedPermutedLanguageModeling | None = None,
        collator_state: dict[str, Any] | None = None,
    ) -> tuple[float, float]:
        """Evaluate a model on a dataloader split.

        :param torch.nn.Module model_to_eval: Model to evaluate.
        :param torch.utils.data.DataLoader dataloader: Dataloader for the split.
        :param str split_name: Display name for logging.
        :param str loss_key: Meter key to store loss values.
        :param str acc_key: Meter key to store accuracy values.
        :param DataCollatorForMaskedPermutedLanguageModeling collator: Optional eval collator.
        :param dict[str, Any] collator_state: Optional collator RNG state for deterministic eval.
        :return tuple[float, float]: Average loss and accuracy for the split.
        """
        meters[loss_key].reset()
        meters[acc_key].reset()
        if collator is not None and collator_state is not None:
            collator.set_rng_state(collator_state)
        model_to_eval.eval()
        for _, batch in track(
            enumerate(dataloader),
            description=f"{split_name} evaluation",
            total=len(dataloader),
        ):
            device_batch = {
                data_type: (t.to(device) if isinstance(t, torch.Tensor) else t)
                for data_type, t in batch.items()
            }
            targets = device_batch["targets"]
            with torch.no_grad():
                with _get_autocast_context(device):
                    outs = model_to_eval(**_select_model_inputs(device_batch))
                logits = outs.float().view(-1, outs.size(-1))
                loss = F.cross_entropy(
                    logits,
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
                    meters[loss_key].update(normal_loss, pred_tokens)
                    meters[acc_key].update(normal_acc, pred_tokens)

        return meters[loss_key].avg, meters[acc_key].avg

    eval_interval_steps = args.eval_interval_steps
    if eval_interval_steps is None:
        eval_interval_steps = args.checkpoint_interval if args.checkpoint_interval > 0 else 5000
    if eval_interval_steps <= 0:
        eval_interval_steps = args.total_updates

    # best_loss is already initialized above (possibly from a checkpoint)
    # Flag to track if non-trainable model repo files were saved at first checkpoint.
    initial_outputs_saved = False
    last_eval_step: int | None = None

    # Container to track current streaming dataset for RNG state serialization.
    # Using a list as a mutable container allows the nested generator to update it.
    current_streaming_dataset: list[HFStreamingDataset | None] = [None]

    def _iter_train_batches() -> Iterator[tuple[dict[str, Any], int, int]]:
        """Yield training batches with cycle and batch index.

        :return Iterator[tuple[dict[str, Any], int, int]]: (batch, cycle, batch_index) tuples.
        """
        if train_streaming:
            cycle = resume_data_state["cycle"]
            skip_samples = resume_cycle_samples if resume_mode == "streaming" else 0
            while True:
                LOGGER.info("Starting streaming shuffle cycle %s", cycle)
                current_stream = train_stream.shuffle(
                    buffer_size=args.buffer_size, seed=args.seed + cycle
                )
                if hasattr(current_stream, "set_epoch"):
                    current_stream.set_epoch(cycle)

                local_skip = skip_samples
                num_workers = args.num_workers if hasattr(args, "num_workers") else 0
                if local_skip > 0 and num_workers > 0:
                    per_worker_skip = local_skip // max(num_workers, 1)
                    if per_worker_skip > 0:
                        local_skip = per_worker_skip
                    else:
                        local_skip = 0
                    LOGGER.info(
                        "Resuming streaming dataset: global skip %s -> per-worker skip %s.",
                        skip_samples,
                        local_skip,
                    )

                if local_skip > 0:
                    LOGGER.info(
                        "Resuming streaming dataset: skipping %s already-processed samples.",
                        local_skip,
                    )
                    # Note: skip_samples applies per worker in HFStreamingDataset, so resume skips are
                    # best-effort when num_workers > 0.

                train_dataset = HFStreamingDataset(
                    tokenizer=tokenizer,
                    dataset_stream=current_stream,
                    block_size=args.max_tokens,
                    buffer_size=args.buffer_size,
                    seed=args.seed + cycle,
                    text_field=args.text_field,
                    skip_samples=local_skip,
                )
                # Track current streaming dataset for RNG state serialization at checkpoint time.
                current_streaming_dataset[0] = train_dataset

                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    collate_fn=train_collator,
                    num_workers=args.num_workers if hasattr(args, "num_workers") else 4,
                )
                skip_samples = 0
                batch_index = 0
                for batch in train_dataloader:
                    batch_index += 1
                    yield batch, cycle, batch_index
                cycle += 1
                del train_dataloader
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            cycle = resume_data_state["cycle"]
            skip_batches = resume_cycle_batch_index if resume_mode == "files" else 0
            while True:
                current_train_file = train_files[cycle % len(train_files)]
                LOGGER.info("Starting file cycle %s using file: %s", cycle, current_train_file)

                epoch_train_dataset = MPNetDataset(
                    tokenizer=tokenizer,
                    file_path=current_train_file,
                    block_size=args.max_tokens,
                )

                sampler = RandomSamplerWithSeed(
                    epoch_train_dataset, epoch=cycle, random_seed=args.seed
                )

                train_dataloader = torch.utils.data.DataLoader(
                    epoch_train_dataset,
                    sampler=sampler,
                    collate_fn=train_collator,
                    batch_size=args.batch_size,
                )

                if skip_batches > 0:
                    LOGGER.info(
                        "Resuming file-based dataset: skipping %s already-processed batches.",
                        skip_batches,
                    )

                batch_index = 0
                for batch in train_dataloader:
                    batch_index += 1
                    if skip_batches > 0:
                        skip_batches -= 1
                        continue
                    yield batch, cycle, batch_index

                cycle += 1
                skip_batches = 0
                del train_dataloader
                del epoch_train_dataset
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    train_iter = _iter_train_batches()
    try:
        scheduler.optimizer.zero_grad()
        model.train()

        accumulation_loss = 0
        accumulation_acc = 0  # Count of correct predictions; normalize by predicted tokens later.
        accumulation_input_tokens = 0
        accumulation_pred_tokens = 0
        micro_steps = 0
        current_cycle = resume_data_state["cycle"]
        cycle_samples_processed = resume_cycle_samples
        cycle_batch_index = resume_cycle_batch_index

        while steps < args.total_updates:
            batch, batch_cycle, batch_index = next(train_iter)
            if batch_cycle != current_cycle:
                current_cycle = batch_cycle
                cycle_samples_processed = 0

            device_batch = {
                data_type: (t.to(device) if isinstance(t, torch.Tensor) else t)
                for data_type, t in batch.items()
            }

            batch_size = int(device_batch["input_ids"].shape[0])
            samples_processed += batch_size
            cycle_samples_processed += batch_size
            cycle_batch_index = batch_index

            targets = device_batch["targets"]
            input_token_count = int(device_batch["ntokens"])
            pred_token_count = _count_pred_tokens(targets, tokenizer.pad_token_id)

            accumulation_input_tokens += input_token_count
            accumulation_pred_tokens += pred_token_count

            with _get_autocast_context(device):
                outs = model(**_select_model_inputs(device_batch))

            # Compute loss in fp32 outside autocast for numerical stability.
            logits = outs.float().view(-1, outs.size(-1))
            loss = F.cross_entropy(
                logits,
                targets.view(-1),
                reduction="sum",
                ignore_index=tokenizer.pad_token_id,
            )

            acc = accuracy(outs, targets, ignore_index=tokenizer.pad_token_id)
            accumulation_acc += acc
            accumulation_loss += loss.item()
            loss.backward()
            micro_steps += 1

            if micro_steps % args.update_freq == 0:
                _scale_gradients_by_tokens(model, accumulation_pred_tokens)

                if args.clip_grad_norm > 0.0:
                    gnorm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_grad_norm
                    ).item()
                else:
                    grad_norm_sq = torch.zeros((), device=device)
                    for p in model.parameters():
                        if p.grad is not None:
                            grad_norm_sq += p.grad.data.norm() ** 2
                    gnorm = grad_norm_sq.sqrt().item()

                # Step the LR for the upcoming update. steps tracks completed updates.
                lr = scheduler.step(steps + 1)
                scheduler.optimizer.step()
                scheduler.optimizer.zero_grad()

                normal_acc = _normalize_training_accuracy(
                    accumulation_acc, accumulation_pred_tokens
                )
                if accumulation_pred_tokens > 0:
                    normal_loss = accumulation_loss / accumulation_pred_tokens / math.log(2)
                else:
                    normal_loss = 0.0

                LOGGER.debug("Accumulated batch information is below:")
                LOGGER.debug(accumulation_pred_tokens)
                LOGGER.debug(accumulation_loss)
                LOGGER.debug(accumulation_input_tokens)

                if accumulation_pred_tokens > 0:
                    meters["train_acc"].update(normal_acc, accumulation_pred_tokens)
                    meters["train_loss"].update(normal_loss, accumulation_pred_tokens)
                meters["token_throughput"].update(accumulation_input_tokens)

                logging_dict = {
                    "acc": meters["train_acc"].avg,
                    "loss": normal_loss,
                    "sbal": meters["train_loss"].avg,
                    "lr": lr,
                    "gnorm": gnorm,
                    "ttp": meters["token_throughput"].sum,
                    "tpb": meters["token_throughput"].avg,
                }

                if args.tensorboard_log_dir is not None:
                    write_to_tensorboard(writers["train"], logging_dict, steps)
                else:
                    LOGGER.info(logging_dict)

                if args.wandb:
                    log_to_wandb(logging_dict, steps, "train")

                accumulation_acc = 0
                accumulation_loss = 0
                accumulation_input_tokens = 0
                accumulation_pred_tokens = 0

                steps += 1

                data_state = {
                    "mode": resume_mode,
                    "cycle": current_cycle,
                    "batch_index": cycle_batch_index,
                    "samples_in_cycle": cycle_samples_processed,
                }

                if _should_save_checkpoint(steps, args.checkpoint_interval):
                    checkpoint = {
                        "args": vars(args),
                        "model_states": model.state_dict(),
                        "steps": steps,
                        "best_loss": best_loss,
                        "samples_processed": samples_processed,
                        "data_state": data_state,
                        "rng_state": {
                            "torch": torch.get_rng_state(),
                            "cuda": (
                                torch.cuda.get_rng_state_all()
                                if torch.cuda.is_available()
                                else None
                            ),
                            "numpy": _serialize_numpy_rng_state(
                                np.random.get_state() if "numpy.random" in sys.modules else None
                            ),
                        },
                        "collator_rng_state": train_collator.get_rng_state(),
                        "streaming_rng_state": None,
                    }

                    checkpoint_path = checkpoint_dir / f"checkpoint{steps}.pt"
                    _atomic_torch_save(checkpoint, checkpoint_path)

                    if args.save_optimizer_state:
                        optimizer_state = {
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "steps": steps,
                            "data_state": data_state,
                        }
                        optimizer_state_path = (
                            optimizer_dir / f"{checkpoint_path.stem}_optimizer_state.pt"
                        )
                        _atomic_torch_save(optimizer_state, optimizer_state_path)

                    if not initial_outputs_saved:
                        args_dict = vars(args) if not isinstance(args, dict) else args
                        args_path = checkpoint_dir / "training_args.json"
                        with open(args_path, "w") as f:
                            json.dump(args_dict, f, indent=4)

                        tokenizer_dir = checkpoint_dir / "tokenizer"
                        tokenizer.save_pretrained(tokenizer_dir)
                        initial_outputs_saved = True

                    _prune_checkpoints(
                        checkpoint_dir,
                        args.keep_checkpoints,
                        optimizer_dir if args.save_optimizer_state else None,
                    )

                if has_validation and eval_interval_steps > 0 and steps % eval_interval_steps == 0:
                    final_valid_loss, final_valid_accuracy = _evaluate_split(
                        model,
                        valid_dataloader,
                        "Validation",
                        "valid_loss",
                        "valid_acc",
                        collator=eval_collator,
                        collator_state=eval_collator_state,
                    )
                    last_eval_step = steps
                    if final_valid_loss < best_loss:
                        best_loss = final_valid_loss
                        best_checkpoint = {
                            "args": vars(args),
                            "model_states": model.state_dict(),
                            "steps": steps,
                            "best_loss": best_loss,
                            "samples_processed": samples_processed,
                            "data_state": data_state,
                            "rng_state": {
                                "torch": torch.get_rng_state(),
                                "cuda": (
                                    torch.cuda.get_rng_state_all()
                                    if torch.cuda.is_available()
                                    else None
                                ),
                                "numpy": _serialize_numpy_rng_state(
                                    np.random.get_state() if "numpy.random" in sys.modules else None
                                ),
                            },
                            "collator_rng_state": train_collator.get_rng_state(),
                            "streaming_rng_state": None,
                        }

                        best_checkpoint_path = checkpoint_dir / "best_checkpoint.pt"
                        _atomic_torch_save(best_checkpoint, best_checkpoint_path)

                        if args.save_optimizer_state:
                            best_optimizer_state = {
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "steps": steps,
                                "best_loss": best_loss,
                                "data_state": data_state,
                            }
                            best_optimizer_state_path = optimizer_dir / "best_optimizer_state.pt"
                            _atomic_torch_save(best_optimizer_state, best_optimizer_state_path)

                    logging_dict = {
                        "loss": final_valid_loss,
                        "acc": final_valid_accuracy,
                        "best_loss": best_loss,
                    }

                    if args.tensorboard_log_dir:
                        write_to_tensorboard(writers["valid"], logging_dict, steps)
                    else:
                        LOGGER.info("Validation stats:")
                        LOGGER.info(logging_dict)

                    if args.wandb:
                        log_to_wandb(logging_dict, steps, "valid")

                    for stat in ["train_loss", "train_acc"]:
                        meters[stat].reset()
                    model.train()

    finally:
        # Ensure streaming/file dataloaders release resources on exit.
        train_iter.close()
    if has_validation and last_eval_step != steps:
        final_valid_loss, final_valid_accuracy = _evaluate_split(
            model,
            valid_dataloader,
            "Validation",
            "valid_loss",
            "valid_acc",
            collator=eval_collator,
            collator_state=eval_collator_state,
        )
        if final_valid_loss < best_loss:
            best_loss = final_valid_loss
            final_data_state = {
                "mode": resume_mode,
                "cycle": current_cycle,
                "batch_index": cycle_batch_index,
                "samples_in_cycle": cycle_samples_processed,
            }
            best_checkpoint = {
                "args": vars(args),
                "model_states": model.state_dict(),
                "steps": steps,
                "best_loss": best_loss,
                "samples_processed": samples_processed,
                "data_state": final_data_state,
                "rng_state": {
                    "torch": torch.get_rng_state(),
                    "cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
                    "numpy": _serialize_numpy_rng_state(
                        np.random.get_state() if "numpy.random" in sys.modules else None
                    ),
                },
                "collator_rng_state": train_collator.get_rng_state(),
                "streaming_rng_state": None,
            }
            best_checkpoint_path = checkpoint_dir / "best_checkpoint.pt"
            _atomic_torch_save(best_checkpoint, best_checkpoint_path)
            if args.save_optimizer_state:
                best_optimizer_state = {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "steps": steps,
                    "best_loss": best_loss,
                    "data_state": final_data_state,
                }
                best_optimizer_state_path = optimizer_dir / "best_optimizer_state.pt"
                _atomic_torch_save(best_optimizer_state, best_optimizer_state_path)

        logging_dict = {
            "loss": final_valid_loss,
            "acc": final_valid_accuracy,
            "best_loss": best_loss,
        }
        if args.tensorboard_log_dir:
            write_to_tensorboard(writers["valid"], logging_dict, steps)
        else:
            LOGGER.info("Validation stats:")
            LOGGER.info(logging_dict)
        if args.wandb:
            log_to_wandb(logging_dict, steps, "valid")

    # If we've reached the end of the training cycle, i.e., hit total number of update steps, we can
    # use the test dataloader we built above to get a final test metric using the best checkpoint.
    if has_test:
        # Begin by loading the model states and args from the best checkpoint.
        # Prefer local best checkpoint; fall back to resume root if none was written.
        # If no best checkpoint exists, use the in-memory model from the final step.
        best_checkpoint_path = _select_best_checkpoint_path(checkpoint_dir, resume_checkpoint_path)
        test_model: torch.nn.Module | None = None
        if best_checkpoint_path.exists():
            try:
                dicts = _safe_torch_load(
                    best_checkpoint_path,
                    map_location="cpu",
                    trust_checkpoint=args.trust_checkpoint,
                )

                # Handle args that might be dict or Namespace
                loaded_args = dicts["args"]
                if isinstance(loaded_args, dict):
                    loaded_args = Namespace(**loaded_args)

                # Handle potential _orig_mod prefix in state dict from compiled models
                model_states = _strip_compile_prefix(dicts["model_states"])

                # Load an empty shell of the model architecture using those args
                test_model = MPNetForPretraining(loaded_args, tokenizer)

                # Now apply the model states to this newly instantiated model
                test_model.load_state_dict(model_states)

                # Finally make sure the model is in eval mode and is sent to the proper device
                test_model.to(device)
            except (OSError, RuntimeError, KeyError, ValueError) as exc:
                LOGGER.warning(
                    "Could not load best checkpoint %s for test evaluation: %s. "
                    "Using in-memory model instead.",
                    best_checkpoint_path,
                    exc,
                )
        else:
            LOGGER.warning(
                "Best checkpoint not found at %s; using in-memory model for test evaluation.",
                best_checkpoint_path,
            )

        if test_model is None:
            test_model = model

        test_model.eval()

        # Reuse the shared evaluation helper to keep test/valid logic consistent.
        final_test_loss, final_test_accuracy = _evaluate_split(
            test_model,
            test_dataloader,
            "Test",
            "test_loss",
            "test_acc",
            collator=eval_collator,
            collator_state=eval_collator_state,
        )

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
    else:
        LOGGER.warning("Skipping final test evaluation because test dataloader is empty.")

    LOGGER.info(
        f"Training is finished! See output in {args.checkpoint_dir} and "
        f"tensorboard logs in {args.tensorboard_log_dir}"
    )

    # Finish wandb run if active
    if args.wandb and wandb is not None and wandb.run is not None:
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
        help="The directory containing training files. Each file is fully loaded into memory per "
        "cycle; for large corpora prefer --dataset-name streaming.",
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
        "will override --train-dir, --valid-file, and --test-file. Defaults to "
        f"'{DEFAULT_STREAMING_DATASET}' when no file-based paths are provided.",
        type=str,
        default=None,
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
        "--eval-interval-steps",
        help="How often (in update steps) to run validation. Defaults to --checkpoint-interval if "
        "set, otherwise 5000.",
        default=None,
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
        "--gradient-checkpointing",
        help="Enable activation checkpointing for encoder layers to reduce memory usage.",
        action="store_true",
        default=False,
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
        "--save_steps",
        "--save-steps",
        "--checkpoint-interval",
        dest="checkpoint_interval",
        help="The number of steps to be taken before saving the model",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--keep-checkpoints",
        help="How many interval checkpoints to keep (-1 disables pruning; 0 keeps none).",
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
        default=0,
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
        help="Whether to resume training from a checkpoint. Full resume requires checkpoints "
        "created by v0.1.5+ (data_state); legacy checkpoints will only initialize weights.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--resume-checkpoint",
        help="Path to the checkpoint to resume from. If not provided, will use best_checkpoint.pt. "
        "If best_checkpoint.pt is missing, the latest interval checkpoint will be used. "
        "Legacy checkpoints will only initialize weights.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--trust-checkpoint",
        help="Allow unsafe checkpoint loading for legacy or external .pt files.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--hf-model-path",
        help="Path to a HuggingFace MPNet model to initialize weights from (alternative to resuming from repo checkpoint)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save-optimizer-state",
        help="Whether to save optimizer state for resumable training (required for full resume).",
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

    # Normalize empty dataset-name for backward compatibility with older commands.
    if args.dataset_name == "":
        args.dataset_name = None

    if args.eval_samples < 0:
        parser.error("--eval-samples must be >= 0.")

    has_any_file = any([args.train_dir, args.valid_file, args.test_file])
    has_file_data = all([args.train_dir, args.valid_file, args.test_file])

    if args.dataset_name is None:
        if has_any_file and not has_file_data:
            parser.error("File-based data requires --train-dir, --valid-file, and --test-file.")
        if not has_file_data:
            args.dataset_name = DEFAULT_STREAMING_DATASET
    elif has_any_file:
        LOGGER.warning("--dataset-name provided; ignoring --train-dir/--valid-file/--test-file.")

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
