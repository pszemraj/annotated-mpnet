"""
Module for integrating PyTorch's FlexAttention with MPNet.
This module provides implementation for using FlexAttention as an alternative
to the existing RelativeMultiHeadAttention in MPNet.
"""

import logging
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


class FlexTwoStreamAttention(nn.Module):
    """
    Implementation of two-stream attention using PyTorch's FlexAttention mechanism.
    This class provides a drop-in replacement for RelativeMultiHeadAttention in MPNet
    while using the more efficient FlexAttention backend.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        max_relative_positions=128,
        use_sliding_window=False,
        sliding_window_size=1024,
    ):
        super().__init__()

        # Store parameters
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.max_relative_positions = max_relative_positions
        self.use_sliding_window = use_sliding_window
        self.sliding_window_size = sliding_window_size

        # Define projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize the weights for the linear layers."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)
            nn.init.constant_(self.k_proj.bias, 0.0)
            nn.init.constant_(self.v_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        attn_mask=None,
        positions_bias=None,
        need_weights=False,
        **kwargs
    ):
        """
        Forward pass for FlexTwoStreamAttention.
        
        Args:
            query: Tensor of shape [seq_len, batch_size, embedding_dim] or tuple of (c, q)
            key: Tensor of shape [seq_len, batch_size, embedding_dim]
            value: Tensor of shape [seq_len, batch_size, embedding_dim]
            key_padding_mask: Mask for padding tokens
            attn_mask: Attention mask
            positions_bias: Relative position bias
            need_weights: Whether to return attention weights
            
        Returns:
            output: Tuple of (c_out, q_out) where each is [seq_len, batch_size, embedding_dim]
            attn_weights: Optional attention weights if need_weights is True
        """
        # For two-stream attention, query is actually a tuple of [c, q]
        # Where c is content stream and q is query stream
        c, q = query if isinstance(query, tuple) else (query, query)
        
        tgt_len, bsz, embed_dim = key.size()
        
        # Project inputs
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Project content and query streams separately
        c_proj = self.q_proj(c)
        q_proj = self.q_proj(q)
        
        # Reshape for multi-head attention
        # We keep track of the original sequence lengths before transposes
        c_len = c_proj.size(0)
        q_len = q_proj.size(0)
        
        # Reshape to [batch_size * num_heads, seq_len, head_dim]
        k = k.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        c_proj = c_proj.view(c_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        q_proj = q_proj.view(q_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        
        # Scale queries
        c_proj = c_proj * self.scaling
        q_proj = q_proj * self.scaling
        
        # Process content stream attention
        c_out = self._process_stream(
            c_proj, k, v, 
            bsz, 
            content_mask=attn_mask, 
            position_bias=positions_bias
        )
        
        # Process query stream attention
        q_out = self._process_stream(
            q_proj, k, v, 
            bsz, 
            content_mask=kwargs.get('query_mask'), 
            position_bias=kwargs.get('query_position_bias', positions_bias)
        )
        
        # Return outputs from both streams
        return (c_out, q_out), None
    
    def _process_stream(
        self,
        query,
        key,
        value,
        batch_size,
        content_mask=None,
        position_bias=None,
    ):
        """
        Process a single attention stream (either content or query) using FlexAttention.
        
        Args:
            query: Projected query tensor
            key: Projected key tensor
            value: Projected value tensor
            batch_size: Batch size
            content_mask: Attention mask for this stream
            position_bias: Position bias for this stream
            
        Returns:
            output: Processed output tensor
        """
        # Create score modification function that incorporates relative position bias
        def score_mod_with_bias(score, b, h, q_idx, kv_idx):
            # Apply position bias if provided
            if position_bias is not None:
                # Extract the bias for the current head and positions
                bias_value = position_bias[b * self.num_heads + h, q_idx, kv_idx]
                return score + bias_value
            return score
        
        # Fix tensor dimensions for FlexAttention: [batch, heads, seq_len, head_dim]
        # Currently: [batch*heads, seq_len, head_dim]
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        seq_len_v = value.size(1)
        
        # Reshape to correct dimensions: [batch, heads, seq_len, head_dim]
        query = query.view(batch_size, self.num_heads, seq_len_q, self.head_dim)
        key = key.view(batch_size, self.num_heads, seq_len_k, self.head_dim)
        value = value.view(batch_size, self.num_heads, seq_len_v, self.head_dim)
        
        # Handle sliding window attention if enabled
        if self.use_sliding_window:
            def sliding_window_mask(b, h, q_idx, kv_idx):
                # For causal masking with sliding window
                causal_mask = q_idx >= kv_idx
                window_mask = q_idx - kv_idx <= self.sliding_window_size
                return causal_mask & window_mask
            
            # Create block mask for sliding window attention
            block_mask = create_block_mask(
                sliding_window_mask, 
                batch_size, 
                self.num_heads, 
                seq_len_q, 
                seq_len_k
            )
            
            # Apply FlexAttention with sliding window mask
            attn_output = flex_attention(
                query,  # [batch, heads, seq_len, head_dim]
                key,    # [batch, heads, seq_len, head_dim]
                value,  # [batch, heads, seq_len, head_dim]
                score_mod=score_mod_with_bias,
                block_mask=block_mask,
            )
        else:
            # Apply standard FlexAttention with content mask
            attn_output = flex_attention(
                query,  # [batch, heads, seq_len, head_dim]
                key,    # [batch, heads, seq_len, head_dim]
                value,  # [batch, heads, seq_len, head_dim]
                score_mod=score_mod_with_bias,
            )
        
        # Reshape output back: [seq_len, batch, embed_dim]
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()
        attn_output = attn_output.view(seq_len_q, batch_size, self.embed_dim)
        
        # Apply output projection
        attn_output = self.out_proj(attn_output)
        
        return attn_output


def encode_flex_two_stream_attention(
    self,
    c: torch.Tensor,
    q: torch.Tensor,
    content_mask: torch.Tensor = None,
    query_mask: torch.Tensor = None,
    content_position_bias: torch.Tensor = None,
    query_position_bias: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This is a drop-in replacement for encode_two_stream_attention that uses FlexAttention.
    It preserves the same interface as the original function but uses the FlexTwoStreamAttention
    implementation.
    
    Args:
        self: Reference to the SentenceEncoderLayer instance
        c: The content stream tensor
        q: The query stream tensor
        content_mask: Attention mask for the content stream
        query_mask: Attention mask for the query stream
        content_position_bias: Position bias for the content stream
        query_position_bias: Position bias for the query stream
    
    Returns:
        Tuple containing the processed content and query tensors
    """
    # Helper function for skip connection and layer norm processing
    def skip_norm_ff_fn(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Inner function that processes normalization, skip connections, and feed-forward net"""
        # Apply dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Skip connection
        x = x + residual
        
        # Layer norm
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)
        
        # Save residual for next skip connection
        residual = x
        
        # Normalize before feed-forward if specified
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        
        # Feed-forward network
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final skip connection
        x = x + residual
        
        # Final layer norm
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        
        return x

    # Save residuals for skip connections
    residual_c = c
    residual_q = q
    
    # Apply layer norm before attention if specified
    c = self.maybe_layer_norm(self.self_attn_layer_norm, c, before=True)
    q = self.maybe_layer_norm(self.self_attn_layer_norm, q, before=True)
    
    # Apply two-stream attention using FlexAttention
    (c, q), _ = self.self_attn(
        (c, q),
        key=c,
        value=c,
        attn_mask=content_mask,
        query_mask=query_mask,
        positions_bias=content_position_bias,
        query_position_bias=query_position_bias,
    )
    
    # Process skip connections and feed-forward network
    c = skip_norm_ff_fn(c, residual_c)
    q = skip_norm_ff_fn(q, residual_q)
    
    return c, q


def make_flex_attention_mask(
    input_ids: torch.Tensor, seq_len: int, pred_size: int, sliding_window_size: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates appropriate attention masks for FlexAttention in MPNet.
    This is a modified version of make_query_and_content_mask that works with FlexAttention.
    
    Args:
        input_ids: Input tensor used to determine device
        seq_len: Sequence length of the content portion
        pred_size: Sequence length of the prediction portion
        sliding_window_size: Size of the sliding window for attention (if used)
        
    Returns:
        Tuple of query_mask and content_mask compatible with FlexAttention
    """
    device = input_ids.device
    
    # Create base causal masks
    def make_query_mask():
        # Create the mask portion (i.e. ones)
        mask = torch.triu(torch.ones(pred_size, pred_size), 0)
        mask = (torch.ones(pred_size, seq_len - pred_size), 1 - mask, mask)
        return torch.cat(mask, dim=-1).eq(0)

    def make_content_mask():
        mask = [
            torch.zeros(seq_len - pred_size, pred_size),
            torch.tril(torch.ones(pred_size, pred_size), 0),
        ]
        mask.append(torch.zeros(pred_size, pred_size))
        mask = torch.cat(mask, dim=0)
        mask = (torch.ones(seq_len + pred_size, seq_len - pred_size), mask, 1 - mask)
        return torch.cat(mask, dim=-1).eq(0)
    
    # Apply sliding window if specified
    if sliding_window_size is not None and sliding_window_size > 0:
        query_mask = make_query_mask()
        content_mask = make_content_mask()
        
        # Add sliding window constraint to both masks
        q_len, k_len = query_mask.size()
        query_indices = torch.arange(q_len, device=device).unsqueeze(1)
        key_indices = torch.arange(k_len, device=device).unsqueeze(0)
        
        # Apply sliding window constraint: |query_pos - key_pos| <= sliding_window_size
        window_mask = (query_indices - key_indices).abs() > sliding_window_size
        
        # Combine with existing masks
        query_mask = query_mask | window_mask
        content_mask = content_mask | window_mask
        
        return query_mask.to(device), content_mask.to(device)
    
    # Return standard masks
    return make_query_mask().to(device), make_content_mask().to(device)


def two_stream_self_attention_factory(use_flex_attention=False, sliding_window_size=None):
    """
    Factory function that returns either the original two_stream_self_attention function
    or a version that uses FlexAttention based on the configuration.
    
    Args:
        use_flex_attention: Whether to use FlexAttention
        sliding_window_size: Size of sliding window for attention (if using FlexAttention)
        
    Returns:
        Appropriate two_stream_self_attention function
    """
    if use_flex_attention:
        # Return a function that uses FlexAttention
        def flex_two_stream_self_attention(
            self,
            query: torch.Tensor,
            key: torch.Tensor = None,
            value: torch.Tensor = None,
            query_mask: torch.Tensor = None,
            content_mask: torch.Tensor = None,
            query_position_bias: torch.Tensor = None,
            content_position_bias: torch.Tensor = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Function that implements two-stream self-attention using FlexAttention.
            """
            # Unpack content and query streams
            c, q = query
            
            # Get dimensions
            bsz, embed_dim = key.size(1), key.size(2)
            
            # Create FlexAttention instance if not already present
            if not hasattr(self, '_flex_attention') or self._flex_attention is None:
                self._flex_attention = FlexTwoStreamAttention(
                    embed_dim=embed_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    use_sliding_window=sliding_window_size is not None,
                    sliding_window_size=sliding_window_size or 0,
                )
                self._flex_attention.to(key.device)
            
            # Forward pass through FlexAttention
            (c_out, q_out), _ = self._flex_attention(
                (c, q),
                key=key,
                value=value,
                attn_mask=content_mask,
                query_mask=query_mask,
                positions_bias=content_position_bias,
                query_position_bias=query_position_bias,
            )
            
            return c_out, q_out
        
        return flex_two_stream_self_attention
    else:
        # Return the original two_stream_self_attention function
        # This would be imported from your existing codebase
        from annotated_mpnet.modeling.mpnet_for_pretraining import two_stream_self_attention
        return two_stream_self_attention