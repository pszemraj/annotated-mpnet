"""
Optimized FlexAttention implementation for MPNet.
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
        self.in_proj_weight = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Initialize parameters - follow the original RelativeMultiHeadAttention pattern
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize the weights for the linear layers."""
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.in_proj_bias, 0.0)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def _in_proj_qkv(self, x):
        """Project input to q, k, v with a single matrix multiply."""
        return self._in_proj(x).chunk(3, dim=-1)

    def _in_proj(self, x, start=0, end=None):
        """Linear projection for q, k, v using slices of the same weight matrix."""
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        if end is None:
            end = self.embed_dim
        return F.linear(x, weight[start:end, :], bias[start:end])

    def in_proj_q(self, x):
        return self._in_proj(x, 0, self.embed_dim)

    def in_proj_k(self, x):
        return self._in_proj(x, self.embed_dim, 2 * self.embed_dim)

    def in_proj_v(self, x):
        return self._in_proj(x, 2 * self.embed_dim, 3 * self.embed_dim)

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
            query: Tuple of (c, q) where c is content stream and q is query stream
            key: Key tensor
            value: Value tensor
            key_padding_mask: Mask for padding tokens
            attn_mask: Attention mask
            positions_bias: Relative position bias
            need_weights: Whether to return attention weights
            
        Returns:
            output: Tuple of (c_out, q_out) where each is [seq_len, batch_size, embed_dim]
            attn_weights: Optional attention weights if need_weights is True
        """
        # For two-stream attention, query is actually a tuple of [c, q]
        c, q = query if isinstance(query, tuple) else (query, query)
        
        # Get dimensions
        tgt_len, bsz, embed_dim = key.size()
        assert embed_dim == self.embed_dim
        
        # Project inputs using slices of the same weight matrix
        k = self.in_proj_k(key)
        v = self.in_proj_v(value)
        c_proj = self.in_proj_q(c)
        q_proj = self.in_proj_q(q)
        
        # Remember original sequence lengths
        c_len = c_proj.size(0)
        q_len = q_proj.size(0)
        
        # Reshape and transpose for multi-head attention
        # [seq_len, batch, embed] -> [batch*heads, seq_len, head_dim]
        def reshape_for_attention(x, seq_len):
            return (
                x.view(seq_len, bsz, self.num_heads, self.head_dim)
                .permute(1, 2, 0, 3)
                .contiguous()
                .view(bsz * self.num_heads, seq_len, self.head_dim)
            )
        
        # Apply reshaping to all tensors
        k = reshape_for_attention(k, tgt_len)
        v = reshape_for_attention(v, tgt_len)
        c_proj = reshape_for_attention(c_proj, c_len)
        q_proj = reshape_for_attention(q_proj, q_len)
        
        # Process content stream
        c_out = self._process_stream(
            c_proj.view(bsz, self.num_heads, c_len, self.head_dim),  # Reshape for FlexAttention
            k.view(bsz, self.num_heads, tgt_len, self.head_dim),
            v.view(bsz, self.num_heads, tgt_len, self.head_dim),
            attn_mask=attn_mask,
            position_bias=positions_bias,
            query_len=c_len
        )
        
        # Process query stream
        q_out = self._process_stream(
            q_proj.view(bsz, self.num_heads, q_len, self.head_dim),  # Reshape for FlexAttention
            k.view(bsz, self.num_heads, tgt_len, self.head_dim),
            v.view(bsz, self.num_heads, tgt_len, self.head_dim),
            attn_mask=kwargs.get('query_mask'),
            position_bias=kwargs.get('query_position_bias', positions_bias),
            query_len=q_len
        )
        
        # Return outputs from both streams
        return (c_out, q_out), None
    
    def _process_stream(
        self,
        query,
        key,
        value,
        attn_mask=None,
        position_bias=None,
        query_len=None,
    ):
        """
        Process a single attention stream using FlexAttention.
        
        Args:
            query: [batch, heads, seq_len, head_dim]
            key: [batch, heads, seq_len, head_dim]
            value: [batch, heads, seq_len, head_dim]
            attn_mask: Attention mask
            position_bias: Position bias tensor
            query_len: Query sequence length
            
        Returns:
            output: Processed output tensor [seq_len, batch, embed_dim]
        """
        batch_size = query.size(0)
        
        # Create more efficient score mod function
        if position_bias is not None:
            # In MPNet, position_bias is already in [bsz*heads, qlen, klen] format
            # so we don't need to reshape it
            
            # Pre-compute indices for faster access during score_mod
            head_indices = torch.arange(self.num_heads, device=position_bias.device)
            head_indices = head_indices.repeat_interleave(batch_size)
            batch_indices = torch.arange(batch_size, device=position_bias.device).repeat(self.num_heads)
            
            # Optimized score_mod with pre-computed indices
            def score_mod_with_bias(score, b, h, q_idx, kv_idx):
                # Since b ranges from 0 to bsz*heads-1, we can directly use it
                # instead of calculating an index
                return score + position_bias[b, q_idx, kv_idx]
            
            score_mod = score_mod_with_bias
        else:
            # No bias, so no-op score_mod
            score_mod = lambda score, b, h, q_idx, kv_idx: score
        
        # Check if we're using sliding window
        if self.use_sliding_window and self.sliding_window_size > 0:
            # Create efficient sliding window mask
            def sliding_window_mask_fn(b, h, q_idx, kv_idx):
                # For bi-directional masking with sliding window
                return abs(q_idx - kv_idx) <= self.sliding_window_size
            
            # Create block mask only once per input shape
            block_mask = create_block_mask(
                sliding_window_mask_fn, 
                batch_size, 
                self.num_heads, 
                query.size(2),  # query_len
                key.size(2)     # key_len
            )
            
            # Apply FlexAttention with optimized parameters
            attn_output = flex_attention(
                query * self.scaling,  # Scale query once instead of in score_mod
                key, 
                value,
                score_mod=score_mod,
                block_mask=block_mask,
            )
        else:
            # Apply standard FlexAttention
            attn_output = flex_attention(
                query * self.scaling,  # Scale query once instead of in score_mod
                key, 
                value,
                score_mod=score_mod,
            )
        
        # Reshape back to [seq_len, batch, embed_dim]
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()
        attn_output = attn_output.view(query_len, batch_size, self.embed_dim)
        
        # Apply output projection
        return self.out_proj(attn_output)

def make_flex_attention_mask(
    input_ids: torch.Tensor, seq_len: int, pred_size: int, sliding_window_size: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates optimized attention masks for FlexAttention in MPNet.
    
    Args:
        input_ids: Input tensor used to determine device
        seq_len: Sequence length of the content portion
        pred_size: Sequence length of the prediction portion
        sliding_window_size: Size of the sliding window for attention
        
    Returns:
        Tuple of query_mask and content_mask compatible with FlexAttention
    """
    device = input_ids.device
    
    # Create masks efficiently (minimize redundant operations)
    # Query mask: upper triangular for the prediction tokens
    query_mask = torch.ones(pred_size, seq_len + pred_size, device=device)
    # Set the diagonal to 0 for the pred_size x pred_size portion
    query_mask[:, seq_len:].triu_(diagonal=1)
    query_mask = query_mask.bool()
    
    # Content mask: lower triangular for the prediction tokens
    content_mask = torch.zeros(seq_len + pred_size, seq_len + pred_size, device=device)
    # Fill in the relevant segments
    content_mask[seq_len:, seq_len:].tril_(diagonal=0)
    content_mask = content_mask.bool()
    
    # Apply sliding window constraint if specified (more efficient implementation)
    if sliding_window_size is not None and sliding_window_size > 0:
        # Calculate position differences once
        q_positions = torch.arange(pred_size, device=device).unsqueeze(1)
        k_positions = torch.arange(seq_len + pred_size, device=device).unsqueeze(0)
        position_diff = (q_positions - k_positions).abs()
        
        # Create window mask once and apply to both masks
        window_mask = position_diff > sliding_window_size
        
        # Add window constraints
        q_window_mask = window_mask[:pred_size]
        query_mask = query_mask | q_window_mask
        
        # For content mask, compute only once for the full size
        content_positions = torch.arange(seq_len + pred_size, device=device).unsqueeze(1)
        ck_positions = torch.arange(seq_len + pred_size, device=device).unsqueeze(0)
        c_position_diff = (content_positions - ck_positions).abs()
        c_window_mask = c_position_diff > sliding_window_size
        
        content_mask = content_mask | c_window_mask
    
    # Invert masks to match PyTorch's attention convention (True = masked out)
    return query_mask, content_mask


def two_stream_self_attention_factory(use_flex_attention=False, sliding_window_size=None):
    """
    Factory function that returns the appropriate implementation of two-stream self-attention.
    
    Args:
        use_flex_attention: Whether to use FlexAttention
        sliding_window_size: Size of sliding window for attention
        
    Returns:
        Appropriate two_stream_self_attention function
    """
    if use_flex_attention:
        # Create a more optimized version
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
            Optimized two-stream self-attention using FlexAttention with better module handling.
            """
            # Unpack content and query streams
            c, q = query
            
            # Get dimensions
            bsz, embed_dim = key.size(1), key.size(2)
            
            # Create FlexAttention module once and store at module level
            # Don't create it inside the layer's forward pass
            flex_attention_attr = '_flex_attention_instance'
            if not hasattr(self, flex_attention_attr):
                # Create a fixed instance outside of the compiled region
                # with torch._dynamo.disallow_in_graph():
                setattr(self, flex_attention_attr, FlexTwoStreamAttention(
                    embed_dim=embed_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    use_sliding_window=sliding_window_size is not None,
                    sliding_window_size=sliding_window_size or 0,
                ).to(key.device))
        
            # Get the cached module
            flex_attention_module = getattr(self, flex_attention_attr)
            
            # Handle device transfer if needed
            # if flex_attention_module.device != key.device:
                # with torch._dynamo.disallow_in_graph():
            flex_attention_module = flex_attention_module.to(key.device)
            setattr(self, flex_attention_attr, flex_attention_module)
            
            # Forward pass with cached module
            (c_out, q_out), _ = flex_attention_module(
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
        # Return the original implementation
        from annotated_mpnet.modeling.mpnet_for_pretraining import two_stream_self_attention
        return two_stream_self_attention


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
    Optimized wrapper for two-stream attention with FlexAttention backend.
    
    Args:
        self: Layer reference
        c: Content stream tensor
        q: Query stream tensor
        content_mask: Mask for content stream
        query_mask: Mask for query stream
        content_position_bias: Position bias for content stream
        query_position_bias: Position bias for query stream
        
    Returns:
        Tuple of processed content and query tensors
    """
    # Pre-process tensors once
    residual_c = c
    residual_q = q
    
    # Apply layer norm if needed
    c = self.maybe_layer_norm(self.self_attn_layer_norm, c, before=True)
    q = self.maybe_layer_norm(self.self_attn_layer_norm, q, before=True)
    
    # Get the appropriate attention function
    if not hasattr(self, '_flex_attention_fn'):
        # Import the factory once
        factory_fn = two_stream_self_attention_factory(
            use_flex_attention=True,
            sliding_window_size=getattr(self, 'sliding_window_size', None)
        )
        self._flex_attention_fn = factory_fn
    
    # Apply attention
    c, q = self._flex_attention_fn(
        self,
        [c, q],
        key=c,
        value=c,
        query_mask=query_mask,
        content_mask=content_mask,
        query_position_bias=query_position_bias,
        content_position_bias=content_position_bias,
    )
    
    # Post-processing function (better to define once and reuse)
    def process_output(x, residual):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + residual
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)
        
        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = x + residual
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x
    
    # Apply post-processing to both streams
    c = process_output(c, residual_c)
    q = process_output(q, residual_q)
    
    return c, q