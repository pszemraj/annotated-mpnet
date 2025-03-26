"""
Utils module exported from fairseq for things we might not need, but it's here anyway
"""

import contextlib
import math
import warnings
from collections import defaultdict
from typing import Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda: 0)

SUPPORTED_ACTIVATIONS = [
    "relu",
    "gelu",
    "gelu_accurate",
    "silu",
    "relu2",
    "tanh",
    "linear",
]


def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__

    # assign a unique ID to each module instance, so that incremental state is
    # not shared across module instances
    if not hasattr(module_instance, "_fairseq_instance_id"):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._fairseq_instance_id = INCREMENTAL_STATE_INSTANCE_ID[
            module_name
        ]

    return "{}.{}.{}".format(module_name, module_instance._fairseq_instance_id, key)


def get_incremental_state(module, incremental_state, key):
    """Helper for getting incremental state for an nn.Module."""
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def set_incremental_state(module, incremental_state, key, value):
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value


def make_positions(tensor, padding_idx, onnx_trace=False):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


def get_activation_fn(activation: str) -> Callable:
    """Returns the activation function corresponding to `activation`"""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "silu":
        return F.silu
    elif activation == "relu2":
        return relu_squared
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise ValueError(
            f"{activation} is not supported. Supported activations:\t{SUPPORTED_ACTIVATIONS}"
        )


def gelu_accurate(x: torch.Tensor) -> torch.Tensor:
    """
    An implementation of "accurate" gelu
    """
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (
        0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
    )


def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    An implementation of gelu
    """
    if hasattr(torch.nn.functional, "gelu"):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def relu_squared(x: torch.Tensor) -> torch.Tensor:
    """
    Applies the relu^2 activation introduced in https://arxiv.org/abs/2109.08668v2
    """
    relu_applied = F.relu(x)
    squared = torch.square(relu_applied)
    return squared


@contextlib.contextmanager
def numpy_seed(seed):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def validate_tokenizer(tokenizer, verbose=False) -> Tuple[bool, Dict]:
    """
    Validates that a tokenizer has all the required attributes and methods
    needed for MPNet pretraining, compatible with both Fast and non-Fast tokenizers.

    Args:
        tokenizer: The tokenizer to validate
        verbose: Whether to print detailed information about validation steps

    Returns:
        bool: True if the tokenizer is valid for MPNet pretraining
        dict: Configuration recommendations for using this tokenizer
    """
    valid = True
    issues = []
    recommendations = {}

    # Check required token attributes
    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        valid = False
        issues.append("Missing pad_token - required in SentenceEncoder initialization")

    if not hasattr(tokenizer, "mask_token") or tokenizer.mask_token is None:
        valid = False
        issues.append("Missing mask_token - required for masked language modeling")

    if not hasattr(tokenizer, "mask_token_id") or tokenizer.mask_token_id is None:
        valid = False
        issues.append("Missing mask_token_id - required in mask_perm method")

    # Check for pad_token_id (important for functionality)
    if not hasattr(tokenizer, "pad_token_id") or tokenizer.pad_token_id is None:
        valid = False
        issues.append("Missing pad_token_id - required for padding operations")

    # Check vocab (either traditional dict or within the tokenizer's backend)
    if not hasattr(tokenizer, "vocab_size") or not isinstance(
        tokenizer.vocab_size, int
    ):
        valid = False
        issues.append("Missing vocab_size - required for model initialization")

    # Check for vocab access (allowing for Fast tokenizer differences)
    has_vocab_access = False
    vocab_dict = None

    if hasattr(tokenizer, "vocab") and isinstance(tokenizer.vocab, dict):
        has_vocab_access = True
        vocab_dict = tokenizer.vocab
    elif hasattr(tokenizer, "get_vocab") and callable(tokenizer.get_vocab):
        try:
            vocab_dict = tokenizer.get_vocab()
            if isinstance(vocab_dict, dict):
                has_vocab_access = True
        except Exception as e:
            if verbose:
                warnings.warn(f"get_vocab() method failed: {e}")

    if not has_vocab_access:
        valid = False
        issues.append("Cannot access vocabulary - required for token mapping")

    # Check all_special_ids attribute
    if not hasattr(tokenizer, "all_special_ids") or not isinstance(
        tokenizer.all_special_ids, list
    ):
        valid = False
        issues.append(
            "Missing all_special_ids list - required for token corruption probabilities"
        )

    # Check required methods
    if not callable(getattr(tokenizer, "__call__", None)):
        valid = False
        issues.append("Missing __call__ method - required for tokenization")

    if not callable(getattr(tokenizer, "pad", None)):
        valid = False
        issues.append("Missing pad method - required for batch processing")

    # Test tokenizer functionality
    try:
        sample_text = "This is a test sentence."
        encoding = tokenizer(
            sample_text, add_special_tokens=True, truncation=True, max_length=10
        )
        # More lenient check - just see if we can access input_ids
        try:
            input_ids = (
                encoding["input_ids"]
                if isinstance(encoding, dict)
                else encoding.input_ids
            )
            if input_ids is None:
                valid = False
                issues.append("Tokenizer does not produce input_ids as required")
        except (KeyError, AttributeError):
            valid = False
            issues.append("Cannot access input_ids from tokenizer output")
    except Exception as e:
        valid = False
        issues.append(f"Tokenizer.__call__ test failed: {str(e)}")

    # Test padding functionality
    try:
        # Create encodings to pad
        encodings = [
            tokenizer("Short text", return_tensors=None),
            tokenizer("This is a longer text", return_tensors=None),
        ]

        # Test padding
        padded = tokenizer.pad(encodings, return_tensors="pt")

        # More lenient check - just see if we can access input_ids
        try:
            padded_ids = (
                padded["input_ids"] if isinstance(padded, dict) else padded.input_ids
            )
            if padded_ids is None:
                valid = False
                issues.append("Padded output does not contain input_ids")
        except (KeyError, AttributeError):
            valid = False
            issues.append("Cannot access input_ids from padded output")
    except Exception as e:
        valid = False
        issues.append(f"Padding test failed: {str(e)}")

    # Check for wordpiece tokenization (require a substantial number of ## tokens)
    MIN_WORDPIECE_TOKENS = (
        100  # Minimum number of ## tokens to consider it a WordPiece tokenizer
    )

    if vocab_dict:
        wordpiece_tokens = [
            t for t in vocab_dict.keys() if isinstance(t, str) and t.startswith("##")
        ]  # TODO: adjust data collator to support non-wordpiece tokenization
        wordpiece_count = len(wordpiece_tokens)

        if wordpiece_count >= MIN_WORDPIECE_TOKENS:
            recommendations["whole_word_mask"] = True
            if verbose:
                print(
                    f"\nDetected WordPiece tokenizer with {wordpiece_count} subword tokens starting with '##'"
                )
                print(
                    "This tokenizer is compatible with MPNet's default whole_word_mask=True setting"
                )
        else:
            recommendations["whole_word_mask"] = False
            if verbose:
                print(
                    f"\nNOTE: tokenizer has only {wordpiece_count} tokens with '##' prefix. "
                    "This is not a standard WordPiece tokenizer compatible with MPNet's whole word masking. "
                    "When creating DataCollatorForMaskedPermutedLanguageModeling, set whole_word_mask=False"
                )
                print(
                    "or modify the data collator to support your tokenizer's word boundary convention"
                )

                # Show a few sample tokens to help identify tokenization scheme
                if vocab_dict:
                    sample_tokens = list(vocab_dict.keys())
                    if len(sample_tokens) > 10:
                        sample_tokens = sample_tokens[:10]
                    print(f"\nSample vocabulary tokens: {sample_tokens}")

    # Check for optimal vocab size
    if hasattr(tokenizer, "vocab_size") and isinstance(tokenizer.vocab_size, int):
        original_vocab_size = tokenizer.vocab_size
        target_vocab_size = ((original_vocab_size + 127) // 128) * 128

        if original_vocab_size != target_vocab_size:
            recommendations["padded_vocab_size"] = target_vocab_size
            if verbose:
                print("\nPerformance recommendation:")
                print(f"  Current vocab_size: {original_vocab_size}")
                print(
                    f"  For optimal GPU performance, pad vocabulary to: {target_vocab_size}"
                )

    # Print validation results
    if verbose:
        if valid:
            print("\nTokenizer validation passed! ✅")
            if recommendations:
                print("\nRecommendations:")
                for k, v in recommendations.items():
                    print(f"  - {k}: {v}")
        else:
            print("\nTokenizer validation failed! ❌")
            print("\nIssues found:")
            for i, issue in enumerate(issues):
                print(f"  {i + 1}. {issue}")

            if recommendations:
                print("\nRecommendations (if issues are fixed):")
                for k, v in recommendations.items():
                    print(f"  - {k}: {v}")

    return valid, recommendations
