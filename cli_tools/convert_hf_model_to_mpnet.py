"""
Script for converting HuggingFace MPNet models to our MPNetForPretraining format.
This is the reverse of convert_pretrained_mpnet_to_hf_model.py.
"""

import argparse
import logging
import os
from argparse import Namespace

from rich.logging import RichHandler

LOG_FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=LOG_FORMAT, datefmt="[%X] ", handlers=[RichHandler()]
)
LOGGER = logging.getLogger(__name__)


import torch
from transformers import AutoTokenizer, MPNetForMaskedLM


def convert_hf_model_to_mpnet(
    hf_model_path: str,
    mpnet_checkpoint_path: str,
    model_config: dict = None,
) -> None:
    """
    Convert a HuggingFace MPNet model to our MPNetForPretraining format.

    Args:
        hf_model_path: Path to HuggingFace model or model identifier
        mpnet_checkpoint_path: Path to save the converted checkpoint
        model_config: Optional configuration for the target model, if not provided,
                      it will use the source model's configuration
    """
    LOGGER.info(f"Loading HuggingFace model from {hf_model_path}")

    # Try PyTorch weights first, fallback to TensorFlow if needed
    try:
        # Note: Some "weights not initialized" warnings are expected and can be ignored
        hf_model = MPNetForMaskedLM.from_pretrained(
            hf_model_path, ignore_mismatched_sizes=True
        )
    except ValueError as e:
        if "pytorch_model.bin" in str(e) and "TensorFlow weights" in str(e):
            LOGGER.info("PyTorch weights not found, loading from TensorFlow weights")
            hf_model = MPNetForMaskedLM.from_pretrained(
                hf_model_path, from_tf=True, ignore_mismatched_sizes=True
            )
        else:
            raise

    hf_config = hf_model.config

    # Log model configuration for debugging
    LOGGER.info(
        f"Loaded model config: hidden_size={hf_config.hidden_size}, "
        f"num_layers={hf_config.num_hidden_layers}, "
        f"attention_heads={hf_config.num_attention_heads}"
    )

    # Create the base args for our model format
    if model_config is None:
        args = Namespace(
            encoder_layers=hf_config.num_hidden_layers,
            encoder_embed_dim=hf_config.hidden_size,
            encoder_ffn_dim=hf_config.intermediate_size,
            encoder_attention_heads=hf_config.num_attention_heads,
            dropout=hf_config.hidden_dropout_prob,
            attention_dropout=hf_config.attention_probs_dropout_prob,
            activation_dropout=hf_config.hidden_dropout_prob,
            activation_fn=hf_config.hidden_act,
            normalize_before=False,
            max_positions=hf_config.max_position_embeddings,  # HF already includes special tokens
            relative_attention_num_buckets=hf_config.relative_attention_num_buckets,
            relative_attention_max_distance=None,
            pad_token_id=hf_config.pad_token_id,
            bos_token_id=hf_config.bos_token_id,
            eos_token_id=hf_config.eos_token_id,
            # Store the HF model's vocab size as padded_vocab_size to prevent re-padding
            original_vocab_size=hf_config.vocab_size,
            padded_vocab_size=hf_config.vocab_size,
        )
    else:
        args = Namespace(**model_config)

    LOGGER.info("Creating MPNetForPretraining model with matching configuration")
    from annotated_mpnet.modeling import MPNetForPretraining

    # Create a tokenizer to initialize the model
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)

    # Check if tokenizer supports whole word masking
    try:
        from annotated_mpnet.utils.utils import validate_tokenizer

        is_valid, details = validate_tokenizer(tokenizer)
        if not is_valid:
            LOGGER.warning(
                f"Tokenizer may not support whole word masking: {details}. "
                "Training might not work as expected."
            )
    except ImportError:
        LOGGER.warning("Could not validate tokenizer, continuing anyway.")

    # Create our model
    model = MPNetForPretraining(args, tokenizer)

    # Create mappings from HF model to our model
    mappings = {}

    # Embedding mappings
    mappings["mpnet.embeddings.word_embeddings.weight"] = (
        "sentence_encoder.embed_tokens.weight"
    )
    mappings["mpnet.embeddings.position_embeddings.weight"] = (
        "sentence_encoder.embed_positions.weight"
    )
    mappings["mpnet.embeddings.LayerNorm.weight"] = (
        "sentence_encoder.emb_layer_norm.weight"
    )
    mappings["mpnet.embeddings.LayerNorm.bias"] = "sentence_encoder.emb_layer_norm.bias"

    # Relative attention bias
    mappings["mpnet.encoder.relative_attention_bias.weight"] = (
        "sentence_encoder.relative_attention_bias.weight"
    )

    # LM head mappings
    mappings["lm_head.dense.weight"] = "lm_head.dense.weight"
    mappings["lm_head.dense.bias"] = "lm_head.dense.bias"
    mappings["lm_head.layer_norm.weight"] = "lm_head.layer_norm.weight"
    mappings["lm_head.layer_norm.bias"] = "lm_head.layer_norm.bias"
    mappings["lm_head.decoder.weight"] = "lm_head.weight"
    mappings["lm_head.decoder.bias"] = "lm_head.bias"

    # Handle each encoder layer
    for i in range(args.encoder_layers):
        # Base prefix for HF model
        hf_prefix = f"mpnet.encoder.layer.{i}."
        # Base prefix for our model
        our_prefix = f"sentence_encoder.layers.{i}."

        # Layer norms
        mappings[f"{hf_prefix}attention.LayerNorm.weight"] = (
            f"{our_prefix}self_attn_layer_norm.weight"
        )
        mappings[f"{hf_prefix}attention.LayerNorm.bias"] = (
            f"{our_prefix}self_attn_layer_norm.bias"
        )
        mappings[f"{hf_prefix}output.LayerNorm.weight"] = (
            f"{our_prefix}final_layer_norm.weight"
        )
        mappings[f"{hf_prefix}output.LayerNorm.bias"] = (
            f"{our_prefix}final_layer_norm.bias"
        )

        # Feed-forward network
        mappings[f"{hf_prefix}intermediate.dense.weight"] = f"{our_prefix}fc1.weight"
        mappings[f"{hf_prefix}intermediate.dense.bias"] = f"{our_prefix}fc1.bias"
        mappings[f"{hf_prefix}output.dense.weight"] = f"{our_prefix}fc2.weight"
        mappings[f"{hf_prefix}output.dense.bias"] = f"{our_prefix}fc2.bias"

        # Output projection
        mappings[f"{hf_prefix}attention.attn.o.weight"] = (
            f"{our_prefix}self_attn.out_proj.weight"
        )
        mappings[f"{hf_prefix}attention.attn.o.bias"] = (
            f"{our_prefix}self_attn.out_proj.bias"
        )

        # Special handling for attention QKV weights
        # HF stores them separately, we combine them into single in_proj weight/bias tensors
        q_weight = hf_model.state_dict()[f"{hf_prefix}attention.attn.q.weight"]
        k_weight = hf_model.state_dict()[f"{hf_prefix}attention.attn.k.weight"]
        v_weight = hf_model.state_dict()[f"{hf_prefix}attention.attn.v.weight"]
        q_bias = hf_model.state_dict()[f"{hf_prefix}attention.attn.q.bias"]
        k_bias = hf_model.state_dict()[f"{hf_prefix}attention.attn.k.bias"]
        v_bias = hf_model.state_dict()[f"{hf_prefix}attention.attn.v.bias"]

        # Combine QKV weights and biases
        combined_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        combined_bias = torch.cat([q_bias, k_bias, v_bias])

        # Add to our model state
        model.state_dict()[f"{our_prefix}self_attn.in_proj_weight"].copy_(
            combined_weight
        )
        model.state_dict()[f"{our_prefix}self_attn.in_proj_bias"].copy_(combined_bias)

    # Now apply all the direct mappings
    for hf_key, our_key in mappings.items():
        if hf_key in hf_model.state_dict() and our_key in model.state_dict():
            hf_tensor = hf_model.state_dict()[hf_key]
            our_tensor = model.state_dict()[our_key]
            
            # Special handling for position embeddings size mismatch
            if our_key == "sentence_encoder.embed_positions.weight":
                if hf_tensor.shape[0] != our_tensor.shape[0]:
                    LOGGER.info(
                        f"Position embeddings size mismatch: HF has {hf_tensor.shape[0]}, "
                        f"our model expects {our_tensor.shape[0]}"
                    )
                    # If our model has more positions (e.g., 514 vs 512), pad the HF tensor
                    if our_tensor.shape[0] > hf_tensor.shape[0]:
                        padding_size = our_tensor.shape[0] - hf_tensor.shape[0]
                        # Initialize extra positions with small random values
                        padding = torch.randn(padding_size, hf_tensor.shape[1]) * 0.02
                        hf_tensor = torch.cat([hf_tensor, padding], dim=0)
                        LOGGER.info(f"Padded HF position embeddings by {padding_size} positions")
                    # If HF has more positions, truncate
                    elif our_tensor.shape[0] < hf_tensor.shape[0]:
                        hf_tensor = hf_tensor[:our_tensor.shape[0]]
                        LOGGER.info(f"Truncated HF position embeddings to {our_tensor.shape[0]} positions")
            
            model.state_dict()[our_key].copy_(hf_tensor)

    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = os.path.dirname(mpnet_checkpoint_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Save the model
    LOGGER.info(f"Saving converted model to {mpnet_checkpoint_path}")
    torch.save(
        {"args": vars(args), "model_states": model.state_dict()},
        mpnet_checkpoint_path,
    )
    LOGGER.info("Conversion completed successfully")


def cli_main():
    """
    Command-line interface for the converter
    """
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace MPNet model to annotated-mpnet format"
    )
    parser.add_argument(
        "--hf-model-path",
        type=str,
        required=True,
        help="Path or name of the HuggingFace model to convert",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=str,
        required=True,
        help="Path where to save the converted model checkpoint",
    )
    args = parser.parse_args()

    convert_hf_model_to_mpnet(
        args.hf_model_path,
        args.output_checkpoint,
    )


if __name__ == "__main__":
    cli_main()
