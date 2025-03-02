"""
Implementation of FlexAttention for MPNet based on PyTorch's FlexAttention API.

This module provides flexible attention mechanisms that allow for custom attention patterns
including sliding window attention.
"""

import logging
from typing import Callable, Optional, Tuple

from rich.logging import RichHandler

LOG_FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=LOG_FORMAT, datefmt="[%X] ", handlers=[RichHandler()]
)
LOGGER = logging.getLogger(__name__)

import math

import torch
import torch.nn.functional as F
from torch import nn

# Default block size for sparse block masks
_DEFAULT_SPARSE_BLOCK_SIZE = 128


class BlockMask:
    """
    Class representing a block mask for sparse attention.
    
    This is used to efficiently represent which blocks of the attention matrix should be computed.
    """
    
    def __init__(self, mask: torch.Tensor, block_size_q: int, block_size_k: int):
        """
        Initialize a BlockMask.
        
        Args:
            mask: Binary tensor indicating which blocks to compute (1) and which to mask (0)
            block_size_q: Block size in the query dimension
            block_size_k: Block size in the key dimension
        """
        self.mask = mask
        self.block_size_q = block_size_q
        self.block_size_k = block_size_k
    
    def __repr__(self):
        # Visualize the block mask using Unicode block characters
        H, W = self.mask.shape[-2:]
        lines = []
        
        for i in range(H):
            line = ""
            for j in range(W):
                if self.mask[..., i, j].item() > 0:
                    line += "██"  # Full block (compute)
                else:
                    line += "  "  # Empty block (masked)
            lines.append(line)
        
        return "\n".join(lines)
    
    def sparsity(self):
        """Return the sparsity percentage (% of blocks that are masked out)"""
        total = self.mask.numel()
        nonzero = self.mask.sum().item()
        return 100.0 * (1.0 - nonzero / total)


def create_block_mask(
    mask_mod: Callable,
    B: int,
    H: int,
    Q_LEN: int,
    KV_LEN: int,
    device: str = "cuda",
    KV_BLOCK_SIZE: Optional[int] = None,
    Q_BLOCK_SIZE: Optional[int] = None,
    _compile: bool = True,
) -> BlockMask:
    """
    Create a block mask for sparse attention.
    
    Args:
        mask_mod: Function that returns a boolean mask for attention
        B: Batch size
        H: Number of heads
        Q_LEN: Query sequence length
        KV_LEN: Key/value sequence length
        device: Device to create the mask on
        KV_BLOCK_SIZE: Block size for key/value dimension
        Q_BLOCK_SIZE: Block size for query dimension
        _compile: Whether to compile the mask creation
        
    Returns:
        BlockMask object representing the sparse attention pattern
    """
    # Use default block sizes if not specified
    if KV_BLOCK_SIZE is None:
        KV_BLOCK_SIZE = _DEFAULT_SPARSE_BLOCK_SIZE
    if Q_BLOCK_SIZE is None:
        Q_BLOCK_SIZE = _DEFAULT_SPARSE_BLOCK_SIZE
        
    # Calculate number of blocks
    num_blocks_kv = math.ceil(KV_LEN / KV_BLOCK_SIZE)
    num_blocks_q = math.ceil(Q_LEN / Q_BLOCK_SIZE)
    
    # Initialize block mask
    block_mask = torch.zeros(B, H, num_blocks_q, num_blocks_kv, device=device)
    
    # For each query block, check if any token in it can attend to any token in each key block
    for q_block in range(num_blocks_q):
        q_start = q_block * Q_BLOCK_SIZE
        q_end = min(q_start + Q_BLOCK_SIZE, Q_LEN)
        
        for kv_block in range(num_blocks_kv):
            kv_start = kv_block * KV_BLOCK_SIZE
            kv_end = min(kv_start + KV_BLOCK_SIZE, KV_LEN)
            
            # Sample a few positions from each block to determine if the block is needed
            # For simplicity, we'll just check the corners and center of the block
            positions_to_check = [
                (q_start, kv_start),  # Top-left
                (q_start, kv_end - 1),  # Top-right
                (q_end - 1, kv_start),  # Bottom-left
                (q_end - 1, kv_end - 1),  # Bottom-right
                ((q_start + q_end) // 2, (kv_start + kv_end) // 2),  # Center
            ]
            
            # Check if any position in the block should be attended to
            for b in range(B):
                for h in range(H):
                    for q_idx, kv_idx in positions_to_check:
                        if q_idx < Q_LEN and kv_idx < KV_LEN:
                            # Call the mask_mod function to determine if this position should be attended to
                            if mask_mod(b, h, q_idx, kv_idx):
                                block_mask[b, h, q_block, kv_block] = 1
                                break  # If any position is valid, mark the whole block as valid
    
    return BlockMask(block_mask, Q_BLOCK_SIZE, KV_BLOCK_SIZE)


def create_mask(
    mask_mod: Callable,
    B: int,
    H: int,
    Q_LEN: int,
    KV_LEN: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Create a full attention mask.
    
    Args:
        mask_mod: Function that returns a boolean mask for attention
        B: Batch size
        H: Number of heads
        Q_LEN: Query sequence length
        KV_LEN: Key/value sequence length
        device: Device to create the mask on
        
    Returns:
        Boolean tensor of shape (B*H, Q_LEN, KV_LEN) with False indicating positions to attend to
        and True indicating positions to mask out
    """
    # Create coordinate matrices for all query and key positions
    q_idx = torch.arange(Q_LEN, device=device)
    kv_idx = torch.arange(KV_LEN, device=device)
    
    # Create a 2D grid of all (q_idx, kv_idx) pairs
    q_idx_grid = q_idx.view(-1, 1).expand(-1, KV_LEN)
    kv_idx_grid = kv_idx.view(1, -1).expand(Q_LEN, -1)
    
    # Initialize mask for all batch items and heads
    mask = torch.zeros(B, H, Q_LEN, KV_LEN, dtype=torch.bool, device=device)
    
    # Apply the mask_mod function to each position
    for b in range(B):
        for h in range(H):
            # Call mask_mod for each position
            valid_positions = mask_mod(b, h, q_idx_grid, kv_idx_grid)
            mask[b, h] = ~valid_positions  # Invert because attention_mask uses True for masked positions
    
    # Reshape to the expected format for attention_mask
    mask = mask.view(B * H, Q_LEN, KV_LEN)
    
    return mask


def flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Optional[Callable] = None,
    block_mask: Optional[BlockMask] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute attention with flexible score modification and masking.
    
    Args:
        query: Query tensor of shape (B, H, Q_LEN, D)
        key: Key tensor of shape (B, H, KV_LEN, D)
        value: Value tensor of shape (B, H, KV_LEN, D)
        score_mod: Function to modify attention scores
        block_mask: Block mask for sparse attention
        scale: Scaling factor for attention scores
        
    Returns:
        Output tensor of shape (B, H, Q_LEN, D)
    """
    # Get dimensions
    B, H, Q_LEN, D = query.shape
    _, _, KV_LEN, _ = key.shape
    
    # Apply scaling
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    
    scaled_query = query * scale
    
    # Compute attention scores
    if block_mask is not None:
        # Implement sparse block attention
        output = torch.zeros_like(query)
        
        # Process each block where attention is needed
        for b in range(B):
            for h in range(H):
                for q_block in range(block_mask.mask.shape[-2]):
                    q_start = q_block * block_mask.block_size_q
                    q_end = min(q_start + block_mask.block_size_q, Q_LEN)
                    
                    if q_start >= Q_LEN:
                        continue
                    
                    for kv_block in range(block_mask.mask.shape[-1]):
                        if block_mask.mask[b, h, q_block, kv_block] == 0:
                            continue  # Skip masked blocks
                            
                        kv_start = kv_block * block_mask.block_size_k
                        kv_end = min(kv_start + block_mask.block_size_k, KV_LEN)
                        
                        if kv_start >= KV_LEN:
                            continue
                        
                        # Extract the relevant parts of the tensors
                        q_block_tensor = scaled_query[b, h, q_start:q_end]
                        k_block_tensor = key[b, h, kv_start:kv_end]
                        v_block_tensor = value[b, h, kv_start:kv_end]
                        
                        # Compute attention scores for this block
                        scores = torch.matmul(q_block_tensor, k_block_tensor.transpose(-1, -2))
                        
                        # Apply score_mod if provided
                        if score_mod is not None:
                            # Create coordinate matrices for this block
                            q_idx = torch.arange(q_start, q_end, device=query.device)
                            kv_idx = torch.arange(kv_start, kv_end, device=query.device)
                            
                            # Expand to create a grid
                            q_idx_grid = q_idx.view(-1, 1).expand(-1, kv_end - kv_start)
                            kv_idx_grid = kv_idx.view(1, -1).expand(q_end - q_start, -1)
                            
                            # Apply score_mod
                            scores = score_mod(scores, b, h, q_idx_grid, kv_idx_grid)
                        
                        # Apply softmax and compute weighted sum
                        attn_weights = F.softmax(scores, dim=-1)
                        block_output = torch.matmul(attn_weights, v_block_tensor)
                        
                        # Add to the output tensor
                        output[b, h, q_start:q_end] = block_output
    else:
        # Standard attention computation
        scores = torch.matmul(scaled_query, key.transpose(-1, -2))
        
        # Apply score_mod if provided
        if score_mod is not None:
            # Create coordinate matrices
            q_idx = torch.arange(Q_LEN, device=query.device)
            kv_idx = torch.arange(KV_LEN, device=query.device)
            
            # Expand to create a grid
            q_idx_grid = q_idx.view(-1, 1).expand(-1, KV_LEN)
            kv_idx_grid = kv_idx.view(1, -1).expand(Q_LEN, -1)
            
            # Apply score_mod for each batch and head
            for b in range(B):
                for h in range(H):
                    scores[b, h] = score_mod(scores[b, h], b, h, q_idx_grid, kv_idx_grid)
        
        # Apply softmax and compute weighted sum
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)
    
    return output


class FlexMultiHeadAttention(nn.Module):
    """
    Multi-head attention with flexible attention patterns using FlexAttention.
    
    This is a drop-in replacement for RelativeMultiHeadAttention that supports custom
    attention patterns through score_mod and block_mask.
    """
    
    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        sliding_window_size=None,
    ):
        super().__init__()
        
        # Store args
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.scaling = self.head_dim ** -0.5
        
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        self.sliding_window_size = sliding_window_size
        
        # Parameters for query, key, value projections
        if self.qkv_same_dim:
            self.in_proj_weight = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        else:
            self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
            self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        
        if bias:
            self.in_proj_bias = nn.Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Optional parameters
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        
        self.add_zero_attn = add_zero_attn
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters with Xavier uniform distribution."""
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
            nn.init.xavier_uniform_(self.q_proj_weight)
        
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
        
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
    
    def _in_proj(self, input, start=0, end=None):
        """Apply projection to a portion of the weight matrix."""
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        
        return F.linear(input, weight, bias)
    
    def in_proj_qkv(self, query):
        """Project query to get query, key, value projections at once."""
        return self._in_proj(query).chunk(3, dim=-1)
    
    def in_proj_q(self, query):
        """Project query to get query projection."""
        if self.qkv_same_dim:
            return self._in_proj(query, end=self.embed_dim)
        else:
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[:self.embed_dim]
            return F.linear(query, self.q_proj_weight, bias)
    
    def in_proj_k(self, key):
        """Project key to get key projection."""
        if self.qkv_same_dim:
            return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)
        else:
            weight = self.k_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[self.embed_dim:2 * self.embed_dim]
            return F.linear(key, weight, bias)
    
    def in_proj_v(self, value):
        """Project value to get value projection."""
        if self.qkv_same_dim:
            return self._in_proj(value, start=2 * self.embed_dim)
        else:
            weight = self.v_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[2 * self.embed_dim:]
            return F.linear(value, weight, bias)
    
    def sliding_window_mask_mod(self, b, h, q_idx, kv_idx):
        """
        Create a sliding window attention mask.
        
        Args:
            b: Batch index
            h: Head index
            q_idx: Query position index tensor
            kv_idx: Key/value position index tensor
            
        Returns:
            Boolean tensor with True for positions within the sliding window
        """
        window_size = self.sliding_window_size
        # Standard causal masking (q_idx >= kv_idx) combined with window constraint
        window_mask = (q_idx >= kv_idx) & (q_idx - kv_idx <= window_size)
        return window_mask
    
    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        incremental_state=None,
        need_weights=False,
        static_kv=False,
        attn_mask=None,
        positions_bias=None,
        use_flex_attention=True,
    ):
        """
        Forward pass for FlexMultiHeadAttention.
        
        Args:
            query: Query tensor (T x B x C)
            key: Key tensor (T x B x C)
            value: Value tensor (T x B x C)
            key_padding_mask: Mask for padding tokens (B x T)
            incremental_state: State for incremental decoding
            need_weights: Whether to return attention weights
            static_kv: Whether to reuse keys and values
            attn_mask: Attention mask (L x S)
            positions_bias: Relative position bias
            use_flex_attention: Whether to use the flex attention mechanism
            
        Returns:
            - Output tensor (T x B x C)
            - Attention weights if needed
        """
        # Extract dimensions
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        
        # Compute query, key, value projections
        if self.self_attention:
            # Self-attention: project query, key, value from the same input
            q, k, v = self.in_proj_qkv(query)
        elif self.encoder_decoder_attention:
            # Encoder-decoder attention: project query from decoder, key/value from encoder
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.in_proj_k(key)
                v = self.in_proj_v(value)
        else:
            # Cross-attention: project query, key, value separately
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        
        # Apply scaling to query
        q = q * self.scaling
        
        # Prepare for multi-head attention
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        
        # Reshape to (B, H, T, D) for flex_attention
        q_flex = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
        k_flex = k.view(bsz, self.num_heads, k.size(1), self.head_dim)
        v_flex = v.view(bsz, self.num_heads, v.size(1), self.head_dim)
        
        if use_flex_attention and self.sliding_window_size is not None:
            # Create sliding window mask if needed
            block_mask = create_block_mask(
                self.sliding_window_mask_mod,
                bsz,
                self.num_heads,
                tgt_len,
                k.size(1),
                device=query.device,
            )
            
            # Apply FlexAttention
            attn = flex_attention(
                q_flex,
                k_flex,
                v_flex,
                block_mask=block_mask,
                scale=None,  # Already applied
            )
            
            # Reshape back to (T, B, C)
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        else:
            # Compute standard attention
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            
            # Apply position bias if provided
            if positions_bias is not None:
                attn_weights += positions_bias
                
            # Apply attention mask if provided
            if attn_mask is not None:
                attn_weights += attn_mask
                
            # Apply key padding mask if provided
            if key_padding_mask is not None:
                attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, -1)
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float("-inf"),
                )
                attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, -1)
            
            # Apply softmax and dropout
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            
            # Compute weighted sum
            attn = torch.bmm(attn_weights, v)
            
            # Reshape back to (T, B, C)
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
            
        # Apply output projection
        attn = self.out_proj(attn)
        
        if need_weights:
            # Average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, -1)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
            return attn, attn_weights
        else:
            return attn, None