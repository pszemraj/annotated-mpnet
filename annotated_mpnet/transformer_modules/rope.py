"""
Implementation of Rotary Position Embeddings (RoPE) for MPNet.

RoPE enhances MPNet by encoding position information directly into the query and key representations
through rotation matrices, which preserves their relative positions when computing dot products.
This makes it particularly suitable for MPNet which uses permutation during training.

Paper reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
https://arxiv.org/abs/2104.09864
"""

import logging
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rich.logging import RichHandler

LOG_FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=LOG_FORMAT, datefmt="[%X] ", handlers=[RichHandler()]
)
LOGGER = logging.getLogger(__name__)


class RotaryPositionEmbedding(nn.Module):
    """
    Implementation of Rotary Position Embeddings (RoPE).
    
    RoPE encodes absolute positions with a rotation matrix that naturally
    incorporates explicit relative position information in attention.
    """
    
    def __init__(
        self, 
        dim: int, 
        max_position_embeddings: int = 512, 
        base: int = 10000,
        scaling_factor: float = 1.0
    ):
        """
        Initialize rotary position embeddings.
        
        Args:
            dim: Dimension of the embeddings (must be divisible by 2)
            max_position_embeddings: Maximum sequence length
            base: Base value for the frequency calculation
            scaling_factor: Factor to scale the frequency (useful for extending context length)
        """
        super().__init__()
        
        if dim % 2 != 0:
            raise ValueError(f"Dimension must be divisible by 2, got {dim}")
        
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        
        # Generate the frequency bands
        # These determine how quickly the rotation happens for each dimension
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        
        # Register as buffer so it moves to the device with the model
        # self.register_buffer("inv_freq", self.inv_freq)
        
        # Cache for the cos and sin values for faster forward passes
        self._cos_cached = None
        self._sin_cached = None
        self._cache_seq_length = -1
    
    def _update_cos_sin_cache(self, seq_length: int, device: torch.device):
        """
        Update the cached cos and sin values for sequence positions.
        
        Args:
            seq_length: The sequence length to calculate positions for
            device: The device to put the tensors on
        """
        # Only recompute if sequence length changed
        if seq_length != self._cache_seq_length:
            self._cache_seq_length = seq_length
            
            # Generate position indices
            positions = torch.arange(seq_length, device=device).float() * self.scaling_factor
            
            # Get the matrix of [seq_len, dim/2]
            freqs = torch.einsum('i,j->ij', positions, self.inv_freq)
            
            # Calculate cos and sin
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().to(device)
            self._sin_cached = emb.sin().to(device)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to query and key tensors.
        
        Args:
            q: Query tensor [batch_size, seq_len, num_heads, head_dim] or [seq_len, batch_size, num_heads, head_dim]
            k: Key tensor with same shape as q
            
        Returns:
            Tuple of query and key tensors with rotary embeddings applied
        """
        # Handle different tensor shapes commonly used in transformers
        seq_first = q.dim() == 4 and q.shape[0] != 1
        
        if seq_first:
            # [batch_size, seq_len, num_heads, head_dim]
            seq_len = q.shape[1]
            device = q.device
            
            # Get or update cached cos and sin values
            self._update_cos_sin_cache(seq_len, device)
            
            cos = self._cos_cached[:seq_len].view(1, seq_len, 1, -1)
            sin = self._sin_cached[:seq_len].view(1, seq_len, 1, -1)
        else:
            # [seq_len, batch_size, num_heads, head_dim]
            seq_len = q.shape[0]
            device = q.device
            
            # Get or update cached cos and sin values
            self._update_cos_sin_cache(seq_len, device)
            
            cos = self._cos_cached[:seq_len].view(seq_len, 1, 1, -1)
            sin = self._sin_cached[:seq_len].view(seq_len, 1, 1, -1)
        
        # Apply rotary embeddings
        return self._apply_rotary(q, k, cos, sin)
    
    def _apply_rotary(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to query and key tensors.
        
        Args:
            q: Query tensor
            k: Key tensor
            cos: Cosine tensor for rotation
            sin: Sine tensor for rotation
            
        Returns:
            Tuple of query and key tensors with rotary embeddings applied
        """
        # Reshape for easier rotation operations
        q_embed_dim = q.shape[-1]
        k_embed_dim = k.shape[-1]
        
        # Split the embedding dimensions for rotation
        q_half = q_embed_dim // 2
        k_half = k_embed_dim // 2
        
        # Split q and k into two parts along last dimension
        q_split = q.reshape(q.shape[:-1] + (2, q_half))
        k_split = k.reshape(k.shape[:-1] + (2, k_half))
        
        # The first half and second half of the embeddings
        q1, q2 = q_split[..., 0, :], q_split[..., 1, :]
        k1, k2 = k_split[..., 0, :], k_split[..., 1, :]
        
        # Apply rotation using the rotation matrix:
        # [cos, -sin]
        # [sin,  cos]
        q_out1 = q1 * cos - q2 * sin
        q_out2 = q2 * cos + q1 * sin
        k_out1 = k1 * cos - k2 * sin
        k_out2 = k2 * cos + k1 * sin
        
        # Concatenate the rotated tensors
        q_out = torch.cat([q_out1, q_out2], dim=-1)
        k_out = torch.cat([k_out1, k_out2], dim=-1)
        
        # Ensure output shape matches input shape exactly
        q_out = q_out.view_as(q)
        k_out = k_out.view_as(k)
        
        return q_out, k_out


# Alternative implementation using einsum for cleaner code (can be more efficient on GPUs)
class EinsumRotaryPositionEmbedding(RotaryPositionEmbedding):
    """
    An alternative implementation of RoPE using torch.einsum for cleaner operations.
    This can sometimes be more efficient on GPUs due to better kernel fusion.
    """
    
    def _apply_rotary(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings using einsum.
        
        Args:
            q: Query tensor
            k: Key tensor
            cos: Cosine tensor for rotation
            sin: Sine tensor for rotation
            
        Returns:
            Tuple of query and key tensors with rotary embeddings applied
        """
        # Extract two-dimensional features for rotation
        q_dim = q.shape[-1]
        k_dim = k.shape[-1]
        
        # Reshape q and k for easier rotation
        q_reshaped = q.reshape(*q.shape[:-1], -1, 2)
        k_reshaped = k.reshape(*k.shape[:-1], -1, 2)
        
        # Create rotation matrices
        zeros = torch.zeros_like(cos)
        cos_sin = torch.cat([
            torch.cat([cos, -sin], dim=-1),
            torch.cat([sin, cos], dim=-1),
        ], dim=-2)
        
        # Apply rotation: [batch, seq, head, d_head/2, 2] @ [seq, 1, 1, 2, 2]
        if q.dim() == 4 and q.shape[0] != 1:  # [batch, seq, head, d_head]
            q_out = torch.einsum('bshdc,sdcc->bshdc', q_reshaped.unsqueeze(-1), cos_sin)
            k_out = torch.einsum('bshdc,sdcc->bshdc', k_reshaped.unsqueeze(-1), cos_sin)
        else:  # [seq, batch, head, d_head]
            q_out = torch.einsum('sbhdc,sdcc->sbhdc', q_reshaped.unsqueeze(-1), cos_sin)
            k_out = torch.einsum('sbhdc,sdcc->sbhdc', k_reshaped.unsqueeze(-1), cos_sin)
        
        # Reshape back to original dimensions
        q_out = q_out.squeeze(-1).reshape(*q.shape)
        k_out = k_out.squeeze(-1).reshape(*k.shape)
        
        return q_out, k_out


def apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standalone function to apply rotary position embeddings to query and key tensors.
    This version handles arbitrary tensor shapes for maximum flexibility.
    
    Args:
        q: Query tensor of shape [..., dim]
        k: Key tensor of shape [..., dim]
        cos: Cosine values for rotation
        sin: Sine values for rotation
        
    Returns:
        Tuple of query and key tensors with rotary embeddings applied
    """
    # Handle various tensor shapes by reshaping to match needed format
    orig_q_shape = q.shape
    orig_k_shape = k.shape
    
    # Get embedding dimension (last dimension)
    dim = q.shape[-1]
    
    # Reshape for rotation
    q_reshaped = q.reshape(*q.shape[:-1], -1, 2)
    k_reshaped = k.reshape(*k.shape[:-1], -1, 2)
    
    # Extract two coordinates for rotation
    q1, q2 = q_reshaped[..., 0], q_reshaped[..., 1]
    k1, k2 = k_reshaped[..., 0], k_reshaped[..., 1]
    
    # Apply rotation using the rotation matrix
    q_out1 = q1 * cos - q2 * sin
    q_out2 = q2 * cos + q1 * sin
    k_out1 = k1 * cos - k2 * sin
    k_out2 = k2 * cos + k1 * sin
    
    # Stack rotated results
    q_out = torch.stack([q_out1, q_out2], dim=-1).flatten(-2)
    k_out = torch.stack([k_out1, k_out2], dim=-1).flatten(-2)
    
    # Reshape back to original shapes
    q_out = q_out.reshape(orig_q_shape)
    k_out = k_out.reshape(orig_k_shape)
    
    return q_out, k_out