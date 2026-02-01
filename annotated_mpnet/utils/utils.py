"""
Utils module exported from fairseq for things we might not need, but it's here anyway
"""

import contextlib
import math
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
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


def _get_full_incremental_state_key(module_instance: nn.Module, key: str) -> str:
    """Build a unique incremental state key for a module instance.

    :param nn.Module module_instance: Module instance used as a key namespace.
    :param str key: State key suffix.
    :return str: Fully-qualified incremental state key.
    """
    module_name = module_instance.__class__.__name__

    # assign a unique ID to each module instance, so that incremental state is
    # not shared across module instances
    if not hasattr(module_instance, "_fairseq_instance_id"):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._fairseq_instance_id = INCREMENTAL_STATE_INSTANCE_ID[module_name]

    return "{}.{}.{}".format(module_name, module_instance._fairseq_instance_id, key)


def get_incremental_state(
    module: nn.Module, incremental_state: Optional[Dict[str, Any]], key: str
) -> Optional[Any]:
    """Fetch incremental state for a module.

    :param nn.Module module: Module instance.
    :param dict incremental_state: Incremental state dictionary.
    :param str key: State key suffix.
    :return object: Incremental state value or None.
    """
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def set_incremental_state(
    module: nn.Module,
    incremental_state: Optional[Dict[str, Any]],
    key: str,
    value: Any,
) -> None:
    """Set incremental state for a module.

    :param nn.Module module: Module instance.
    :param dict incremental_state: Incremental state dictionary.
    :param str key: State key suffix.
    :param object value: State value to store.
    """
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value


def make_positions(
    tensor: torch.Tensor, padding_idx: int, onnx_trace: bool = False
) -> torch.Tensor:
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.

    :param torch.Tensor tensor: Input tensor of token IDs.
    :param int padding_idx: Padding index.
    :param bool onnx_trace: Whether ONNX tracing is enabled, defaults to False.
    :return torch.Tensor: Tensor of position indices.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


def get_activation_fn(activation: str) -> Callable:
    """Return the activation function corresponding to ``activation``.

    :param str activation: Activation function name.
    :return Callable: Activation function.
    """
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
    """Compute the accurate GELU activation.

    :param torch.Tensor x: Input tensor.
    :return torch.Tensor: Activated tensor.
    """
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return 0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Compute the GELU activation.

    :param torch.Tensor x: Input tensor.
    :return torch.Tensor: Activated tensor.
    """
    if hasattr(torch.nn.functional, "gelu"):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def relu_squared(x: torch.Tensor) -> torch.Tensor:
    """Apply the ReLU^2 activation.

    :param torch.Tensor x: Input tensor.
    :return torch.Tensor: Activated tensor.
    """
    relu_applied = F.relu(x)
    squared = torch.square(relu_applied)
    return squared


@contextlib.contextmanager
def numpy_seed(seed: Optional[int]) -> Iterator[None]:
    """Context manager to temporarily seed NumPy's PRNG.

    :param int seed: Seed value, defaults to None.
    :return Iterator[None]: Context manager iterator.
    """
    if seed is None:
        yield
        return
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def validate_tokenizer(tokenizer: Any, verbose: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """Validate tokenizer compatibility with MPNet pretraining.

    :param object tokenizer: Tokenizer to validate.
    :param bool verbose: Whether to print detailed validation output, defaults to False.
    :return Tuple[bool, Dict[str, Any]]: Validity flag and recommendations.
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
    if not hasattr(tokenizer, "vocab_size") or not isinstance(tokenizer.vocab_size, int):
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
    if not hasattr(tokenizer, "all_special_ids") or not isinstance(tokenizer.all_special_ids, list):
        valid = False
        issues.append("Missing all_special_ids list - required for token corruption probabilities")

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
        encoding = tokenizer(sample_text, add_special_tokens=True, truncation=True, max_length=10)
        # More lenient check - just see if we can access input_ids
        try:
            input_ids = encoding["input_ids"] if isinstance(encoding, dict) else encoding.input_ids
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
            padded_ids = padded["input_ids"] if isinstance(padded, dict) else padded.input_ids
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
    MIN_WORDPIECE_TOKENS = 100  # Minimum number of ## tokens to consider it a WordPiece tokenizer

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
                print(f"  For optimal GPU performance, pad vocabulary to: {target_vocab_size}")

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


def model_summary(model: nn.Module, max_depth: int = 4, show_input_size: bool = False) -> None:
    """
    Prints an accurate summary of the model, avoiding double-counting of parameters.

    :param model: torch model to summarize
    :param int max_depth: maximum depth of the model to print, defaults to 4
    :param bool show_input_size: whether to show input size for each layer, defaults to False
    """

    def format_params(num_params: int) -> str:
        """Format a parameter count with commas.

        :param int num_params: Number of parameters.
        :return str: Formatted parameter count.
        """
        return f"{num_params:,}" if num_params > 0 else "--"

    def format_size(size: Optional[List[int]]) -> str:
        """Format a shape list into a compact string.

        :param list size: Shape list.
        :return str: Formatted shape string.
        """
        return "x".join(str(x) for x in size) if size else "N/A"

    def count_parameters(module: nn.Module) -> Tuple[int, int]:
        """Count total and trainable parameters for a module.

        :param nn.Module module: Module to inspect.
        :return Tuple[int, int]: Total and trainable parameter counts.
        """
        total_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total_params, trainable_params

    def recursive_summarize(
        module: nn.Module, depth: int, idx: List[int], prefix: str = ""
    ) -> List[Tuple[str, int, int, int, Optional[List[int]], nn.Module]]:
        """Recursively build a summary list for the module tree.

        :param nn.Module module: Module to summarize.
        :param int depth: Current recursion depth.
        :param list idx: Index path of the module.
        :param str prefix: Name prefix for formatting, defaults to "".
        :return list: Summary entries for the module tree.
        """
        summary = []

        total_params, trainable_params = count_parameters(module)

        if depth <= max_depth:
            layer_name = f"{prefix}{type(module).__name__}"
            param_shape = next(
                (p.shape for p in module.parameters(recurse=False) if p.requires_grad),
                None,
            )
            summary.append((layer_name, depth, total_params, trainable_params, param_shape, module))

            for i, (name, child) in enumerate(module.named_children(), 1):
                child_summary = recursive_summarize(child, depth + 1, idx + [i], prefix + "  ")
                summary.extend(child_summary)

        return summary

    summary = recursive_summarize(model, 1, [1])

    max_name_length = max(len(name) for name, _, _, _, _, _ in summary)
    max_shape_length = max(len(format_size(shape)) for _, _, _, _, shape, _ in summary)

    print("=" * (max_name_length + 50))
    header = f"{'Layer (type:depth-idx)':<{max_name_length}} {'Output Shape':>{max_shape_length}} {'Param #':>12} {'Trainable':>10}"
    print(header)
    print("=" * (max_name_length + 50))

    for name, depth, num_params, trainable_params, shape, _ in summary:
        shape_str = format_size(shape) if show_input_size else ""
        print(
            f"{name:<{max_name_length}} {shape_str:>{max_shape_length}} {format_params(num_params):>12} {str(trainable_params > 0):>10}"
        )

    total_params, trainable_params = count_parameters(model)
    print("=" * (max_name_length + 50))
    print(f"Total params: {format_params(total_params)}")
    print(f"Trainable params: {format_params(trainable_params)}")
    print(f"Non-trainable params: {format_params(total_params - trainable_params)}")
    print("=" * (max_name_length + 50))
