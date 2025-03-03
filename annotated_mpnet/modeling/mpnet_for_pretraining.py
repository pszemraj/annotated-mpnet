"""
Module containing the necessary classes for MPNet pretraining with torch.compile compatibility
"""

import logging
from typing import Tuple, Optional, Dict, Any, Union

from rich.logging import RichHandler

LOG_FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=LOG_FORMAT, datefmt="[%X] ", handlers=[RichHandler()]
)
LOGGER = logging.getLogger(__name__)


import torch
import torch.nn.functional as F
from torch import nn

from annotated_mpnet.transformer_modules import LayerNorm, SentenceEncoder
from annotated_mpnet.utils import utils


def init_final_params(module: nn.Module) -> None:
    """
    This is a function that does the final initialization of weights as according to the original
    BERT paper. This is very important to make sure all biases are zeroed out to start and that
    embedding and linear layers start within a normal distribution at instantiation.

    Args:
        module: this is a module within the model. We will use nn.Module's builtin `apply` function
            to apply this as a callable to all submodules
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


class MPNetForPretraining(nn.Module):
    """
    Class containing all the methods required for pretraining MPNet
    """

    def __init__(self, args, tokenizer) -> None:
        super().__init__()
    
        # Check for FlexAttention configuration
        self.use_flex_attention = getattr(args, 'use_flex_attention', False)
        self.sliding_window_size = getattr(args, 'sliding_window_size', None)

        # Let's define the encoder here
        self.args = args
        self.sentence_encoder = SentenceEncoder(
            padding_idx=tokenizer.vocab[tokenizer.pad_token],
            vocab_size=tokenizer.vocab_size,
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_positions,
            num_segments=0,
            encoder_normalize_before=True,
            activation_fn=args.activation_fn,
            normalize_before=args.normalize_before,
            use_flex_attention=self.use_flex_attention,
            sliding_window_size=self.sliding_window_size,
        )

        # Add the language modeling head so that we can do pretraining
        self.lm_head = MPNetLMHead(
            embed_dim=args.encoder_embed_dim,
            output_dim=tokenizer.vocab_size,
            activation_fn=args.activation_fn,
            weight=self.sentence_encoder.embed_tokens.weight,
        )

        # Finally initialize the weights according to the guidelines in the original BERT paper
        self.apply(init_final_params)

    def output_layer(
        self, features: torch.Tensor, masked_tokens: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Wrapper function for language modeling output layer
        """
        return self.lm_head(features, masked_tokens)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        positions: torch.Tensor, 
        pred_size: int, 
        return_mlm: bool = False, 
        **kwargs
    ) -> torch.Tensor:
        """
        Forward function for computing MPNet with FlexAttention support
        
        Args:
            input_ids: Input token IDs
            positions: Position indices
            pred_size: Size of prediction portion
            return_mlm: Whether to return MLM loss
            
        Returns:
            Model output or tuple of outputs if return_mlm is True
        """
        # Calculate initial embeddings
        emb = self.encode_emb(self.sentence_encoder, input_ids, positions)

        # Reverse the tensor for easier extraction
        x = self._reverse_tensor(emb)

        # Separate out content and query streams
        c, q = self._split_tensor(x, pred_size)

        # Get the content and query position biases
        content_position_bias = self.encode_relative_emb(
            self.sentence_encoder, positions[:, :-pred_size]
        )
        query_position_bias = content_position_bias[:, -pred_size:].contiguous()

        # Get the sz of the inital src_length without the tokens to be predicted
        sz = c.size(0) - pred_size

        # Create query and content masks
        if self.use_flex_attention:
            # Import here to avoid circular imports
            from annotated_mpnet.transformer_modules.flex_attention import make_flex_attention_mask
            query_mask, content_mask = make_flex_attention_mask(
                input_ids, 
                sz, 
                pred_size, 
                self.sliding_window_size
            )
        else:
            # Use the original mask creation function
            query_mask, content_mask = self._make_query_and_content_mask(input_ids, sz, pred_size)

        # Process through layers with appropriate attention mechanism
        if self.use_flex_attention:
            # Import the necessary function for FlexAttention
            from annotated_mpnet.transformer_modules.flex_attention import encode_flex_two_stream_attention
            
            # Process through layers with FlexAttention
            for i, layer in enumerate(self.sentence_encoder.layers):
                c, q = encode_flex_two_stream_attention(
                    layer,
                    c,
                    q,
                    content_mask=content_mask,
                    query_mask=query_mask,
                    content_position_bias=content_position_bias,
                    query_position_bias=query_position_bias,
                )
        else:
            # Use standard attention
            for i, layer in enumerate(self.sentence_encoder.layers):
                c, q = self._encode_two_stream_attention(
                    layer,
                    c,
                    q,
                    content_mask,
                    query_mask,
                    content_position_bias,
                    query_position_bias,
                )

        # Process the final layer norm
        q = self.maybe_final_norm(self.sentence_encoder, q)

        # Re-reverse the tensor so we can have it back in the correct format
        q = self._reverse_tensor(q)

        # Project the attention features out to the vocab size for masked token classification
        x = self.output_layer(q)

        # If we also want MLM loss, handle the additional output
        if return_mlm is True:
            c = c[-pred_size:]
            c = self.maybe_final_norm(self.sentence_encoder, c)
            c = self._reverse_tensor(c)
            c = self.output_layer(c)
            return x, c

        return x

    # Helper methods with clear inputs/outputs for better torch.compile compatibility
    @staticmethod
    def _reverse_tensor(x: torch.Tensor) -> torch.Tensor:
        """Reverses a tensor by transposing dimensions 0 and 1"""
        return x.transpose(0, 1)

    @staticmethod
    def _split_tensor(x: torch.Tensor, split_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Splits a tensor into content and query portions"""
        sz = x.size(0) - split_size
        return x[:sz].contiguous(), x[sz:].contiguous()

    @staticmethod
    def _make_query_and_content_mask(
        input_ids: torch.Tensor, seq_len: int, pred_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Creates query and content masks for attention"""
        device = input_ids.device

        # Create query mask
        query_mask = torch.zeros(pred_size, seq_len + pred_size, device=device)
        query_mask[:, :seq_len] = 1
        triu_mask = torch.triu(torch.ones(pred_size, pred_size, device=device), diagonal=1)
        query_mask[:, seq_len:] = 1 - triu_mask
        query_mask = query_mask.eq(0)

        # Create content mask
        content_mask = torch.zeros(seq_len + pred_size, seq_len + pred_size, device=device)
        content_mask[seq_len:, seq_len:] = torch.tril(
            torch.ones(pred_size, pred_size, device=device), diagonal=0
        )
        content_mask = torch.cat([
            torch.ones(seq_len + pred_size, seq_len - pred_size, device=device),
            content_mask,
            1 - content_mask
        ], dim=1)
        content_mask = content_mask.eq(0)

        return query_mask, content_mask

    @staticmethod
    def _encode_two_stream_attention(
        layer,
        c: torch.Tensor,
        q: torch.Tensor,
        content_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        content_position_bias: Optional[torch.Tensor] = None,
        query_position_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stable implementation of two-stream attention for torch.compile compatibility"""
        # Define skip-connection and normalization helper
        def skip_norm_ff(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
            x = F.dropout(x, p=layer.dropout, training=layer.training)
            x = x + residual
            x = layer.maybe_layer_norm(layer.self_attn_layer_norm, x, after=True)
            residual = x
            x = layer.maybe_layer_norm(layer.final_layer_norm, x, before=True)
            x = layer.activation_fn(layer.fc1(x))
            x = F.dropout(x, p=layer.activation_dropout, training=layer.training)
            x = layer.fc2(x)
            x = F.dropout(x, p=layer.dropout, training=layer.training)
            x = x + residual
            x = layer.maybe_layer_norm(layer.final_layer_norm, x, after=True)
            return x

        # Save residuals for skip connections
        residual_c = c
        residual_q = q

        # Apply layer norm if needed
        c = layer.maybe_layer_norm(layer.self_attn_layer_norm, c, before=True)
        q = layer.maybe_layer_norm(layer.self_attn_layer_norm, q, before=True)

        # Process through two-stream attention
        # This is a simplified version of two_stream_self_attention for better compatibility
        # Get dimensions
        bsz, embed_dim = c.size(1), c.size(2)
        
        # Project queries, keys, and values
        k = layer.self_attn.in_proj_k(c)
        v = layer.self_attn.in_proj_v(c)
        q_proj = layer.self_attn.in_proj_q(q)
        c_proj = layer.self_attn.in_proj_q(c)
        
        # Reshape for attention calculation
        head_dim = embed_dim // layer.self_attn.num_heads
        scaling = head_dim ** -0.5
        
        def reshape_for_attention(x):
            return x.contiguous().view(-1, bsz * layer.self_attn.num_heads, head_dim).transpose(0, 1)
        
        k = reshape_for_attention(k)
        v = reshape_for_attention(v)
        q_proj = reshape_for_attention(q_proj) * scaling
        c_proj = reshape_for_attention(c_proj) * scaling
        
        # Apply attention with masks and biases
        def apply_attention(query, mask, bias):
            # Calculate attention weights
            attn_weights = torch.bmm(query, k.transpose(1, 2))
            
            # Apply position bias if provided
            if bias is not None:
                attn_weights += bias
                
            # Apply mask if provided
            if mask is not None:
                attn_weights = attn_weights.masked_fill(
                    mask.unsqueeze(0),
                    float("-inf"),
                )
                
            # Apply softmax and dropout
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = F.dropout(attn_weights, p=layer.self_attn.dropout, training=layer.training)
            
            # Get attention output
            attn = torch.bmm(attn_weights, v)
            attn = attn.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
            return layer.self_attn.out_proj(attn)
        
        # Process content and query streams
        c_out = apply_attention(c_proj, content_mask, content_position_bias)
        q_out = apply_attention(q_proj, query_mask, query_position_bias)
        
        # Apply skip connections and feed-forward
        c = skip_norm_ff(c_out, residual_c)
        q = skip_norm_ff(q_out, residual_q)
        
        return c, q

    # We define some class static methods here that will be used quite a bit across the board
    @staticmethod
    def encode_emb(
        self, input_ids: torch.Tensor, positions: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Method for embedding the input tokens (i.e. input_ids)

        Args:
            input_ids: the input IDs for the given batch
            positions: the position values
        """

        # Use the embedding layer of the sentence encoder to embed these (passed in via the self
        # arg)
        x = self.embed_tokens(input_ids)

        # Scale the embeddings if necessary
        if self.embed_scale is not None:
            x *= self.embed_scale

        # Add in positions
        if positions is not None:
            x += F.embedding(
                positions + 2, self.embed_positions.weight, self.padding_idx
            )

        # Do layer norm
        if self.emb_layer_norm is not None and not self.normalize_before:
            x = self.emb_layer_norm(x)

        # Process dropout
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    @staticmethod
    def maybe_final_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Another helper function to process the final layer norm if necessary
        """
        if self.emb_layer_norm is not None and self.normalize_before:
            return self.emb_layer_norm(x)
        return x

    @staticmethod
    def encode_relative_emb(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Helper function to properly handle relative position bias
        """
        qlen, klen = positions.size(1), positions.size(1)
        context_position = positions[:, :, None]
        memory_position = positions[:, None, :]

        relative_position = memory_position - context_position

        rp_bucket = self.relative_position_bucket(
            relative_position,
            num_buckets=self.relative_attention_num_buckets,
        )
        rp_bucket = rp_bucket.to(positions.device)
        values = self.relative_attention_bias(rp_bucket)
        values = values.permute(0, 3, 1, 2).contiguous()  # [bsz, head, qlen, klen]
        values = values.view(-1, qlen, klen)
        return values


class MPNetLMHead(nn.Module):
    """
    Head for language modeling on the output of MPNet
    """

    def __init__(
        self, embed_dim: int, output_dim: int, activation_fn: str, weight=None
    ) -> None:
        """
        Let's talk about these args so we can better understand what's happening in the LM head

        Args:
            embed_dim: the embedding dimension of the encoder model (usually 768)
            output_dim: the dimension that we want to project out to (usually the vocab size)
            activation_fn: the activation to be using within the LM head
        """
        super().__init__()

        # Let's define the layers for the LM head. It's a pretty simple pipeline

        # Dense FC layer before casting the embed to the vocab size
        self.dense = nn.Linear(embed_dim, embed_dim)

        # Activation function
        self.activation_fn = utils.get_activation_fn(activation_fn)

        # Get the layer norm
        self.layer_norm = LayerNorm(embed_dim)

        # If we don't provide our own weights, we need to initialize them
        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight

        # Finally create the bias layer
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None):
        """
        Forward pass for the LM head

        Args:
            features: outputs of the encoder portion
            masked_tokens: which tokens are masked
        """

        # Only project the unmasked tokens while training, saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        # Step through the network
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)

        # Project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias

        return x