"""
Defining relative multihead attention for MPNet
"""

import logging
from typing import Any, Dict, Optional, Tuple

from rich.logging import RichHandler

LOG_FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=LOG_FORMAT, datefmt="[%X] ", handlers=[RichHandler()])
LOGGER = logging.getLogger(__name__)


import torch
import torch.nn.functional as F
from torch import nn

from annotated_mpnet.utils import utils


class RelativeMultiHeadAttention(nn.Module):
    """
    Handler class for relative multihead attention used in MPNet
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        self_attention: bool = False,
        encoder_decoder_attention: bool = False,
        max_relative_positions: int = 128,
    ) -> None:
        """Initialize relative multi-head attention.

        :param int embed_dim: Embedding dimension.
        :param int num_heads: Number of attention heads.
        :param int kdim: Key dimension, defaults to None.
        :param int vdim: Value dimension, defaults to None.
        :param float dropout: Dropout probability, defaults to 0.0.
        :param bool bias: Whether to use bias terms, defaults to True.
        :param bool add_bias_kv: Whether to add bias to K/V, defaults to False.
        :param bool add_zero_attn: Whether to add zero attention, defaults to False.
        :param bool self_attention: Whether this is self-attention, defaults to False.
        :param bool encoder_decoder_attention: Whether encoder-decoder attention, defaults to False.
        :param int max_relative_positions: Maximum relative positions, defaults to 128.
        """
        super().__init__()

        # Store args
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        # Store bool that tells the class whether Q, K, V matrices are the same dimension
        self.qkv_same_dim = (self.kdim == embed_dim) and (self.vdim == embed_dim)

        # Store more args
        self.num_heads = num_heads
        self.dropout = dropout

        # Integer division to make sure the number of heads is a factor of embed dimension
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and value to be of the same size"
        )

        # Define the attention parameters based on the args above
        if self.qkv_same_dim:
            self.in_proj_weight = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        else:
            self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
            self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))

        # Add bias if it's been specified
        if bias:
            self.in_proj_bias = nn.Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)

        # Add the output layer
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Add bias for K and V matrices if arg specifies it
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        # Not useful for us, but is used down below, so we set it anyway
        self.onnx_trace = False

        self.enable_torch_version = False
        if hasattr(F, "multi_head_attention_forward"):
            self.enable_torch_version = True
        else:
            self.enable_torch_version = False

        self.reset_parameters()

    def prepare_for_onnx_export_(self) -> None:
        """Prepare the module for ONNX export."""
        self.onnx_trace = True

    def reset_parameters(self) -> None:
        """Initialize all parameters."""
        # Init weights for Q, K, and V
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
            nn.init.xavier_uniform_(self.q_proj_weight)

        # Init weight for out layer
        nn.init.xavier_uniform_(self.out_proj.weight)

        # Init bias if it's been specified
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Any]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        positions_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute multi-head relative attention.

        Input shape: Time x Batch x Channel
        Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        :param torch.Tensor query: Query tensor.
        :param torch.Tensor key: Key tensor, defaults to None.
        :param torch.Tensor value: Value tensor, defaults to None.
        :param torch.Tensor key_padding_mask: Key padding mask, defaults to None.
        :param dict incremental_state: Incremental state buffer, defaults to None.
        :param bool need_weights: Whether to return attention weights, defaults to True.
        :param bool static_kv: Whether key/value are static, defaults to False.
        :param torch.Tensor attn_mask: Attention mask, defaults to None.
        :param torch.Tensor positions_bias: Position bias tensor, defaults to None.
        :return Tuple[torch.Tensor, Optional[torch.Tensor]]: Output and optional attention weights.
        """
        # Unpack the dimensions of the query and assert that they match what we expect
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        # Do branching for the type of attention we're doing here
        if self.self_attention:
            # self-attention
            q, k, v = self.in_proj_qkv(query)

        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.in_proj_k(key)
                v = self.in_proj_v(key)

        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)

        # Get scaled Q before performing energy calculation
        q *= self.scaling

        # If bias has been specified on the K, V matrices, process it here
        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        # Do the matrix manipulation for the energy calculation, namely a transpose
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # Not entirely sure what this does, but leaving it in here in case MPNet uses it
        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                prev_key = saved_state["prev_key"].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if "prev_value" in saved_state:
                prev_value = saved_state["prev_value"].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        # Get the source sequence length
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.shape == torch.Size([]):
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask),
                    ],
                    dim=1,
                )

        # Extract the attention weights, i.e., do the energy calculation (QK) before inputting to
        # softmax below
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        # Make sure everything still looks good
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        # Factor in the attention mask now
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        # Factor in the key padding mask
        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if self.onnx_trace:
                attn_weights = torch.where(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    torch.Tensor([float("-Inf")]),
                    attn_weights.float(),
                ).type_as(attn_weights)
            else:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float("-inf"),
                )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # Add in the position bias here
        if positions_bias is not None:
            attn_weights += positions_bias

        # Softmax on the energy calculation
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)

        # Calculate dropout on the softmaxed energy calculation
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Finally calculate the attention using the energy softmax and V
        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        # Run the final attention number through the FC linear layer
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def _in_proj(
        self, input: torch.Tensor, start: int = 0, end: Optional[int] = None
    ) -> torch.Tensor:
        """Project input using the shared in-projection weights.

        :param torch.Tensor input: Input tensor.
        :param int start: Start index in the projection weight, defaults to 0.
        :param int end: End index in the projection weight, defaults to None.
        :return torch.Tensor: Projected tensor.
        """
        weight = self.in_proj_weight
        bias = self.in_proj_bias

        # Adjust the weight based on indices provided in start and end
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]

        # Return the output of a FC linear layer using these selected weights and biases
        return F.linear(input, weight, bias)

    def in_proj_qkv(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project query into Q, K, V tensors.

        :param torch.Tensor query: Query tensor.
        :return Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Q, K, V projections.
        """
        # Use the chunking function to split the tensor out into the Q, K, V components
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_q(self, query: torch.Tensor) -> torch.Tensor:
        """Project query tensor for Q.

        :param torch.Tensor query: Query tensor.
        :return torch.Tensor: Projected Q tensor.
        """
        if self.qkv_same_dim:
            return self._in_proj(query, end=self.embed_dim)
        else:
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[: self.embed_dim]
            return F.linear(query, self.q_proj_weight, bias)

    def in_proj_k(self, key: torch.Tensor) -> torch.Tensor:
        """Project key tensor for K.

        :param torch.Tensor key: Key tensor.
        :return torch.Tensor: Projected K tensor.
        """
        if self.qkv_same_dim:
            return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)
        else:
            weight = self.k_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[self.embed_dim : 2 * self.embed_dim]
            return F.linear(key, weight, bias)

    def in_proj_v(self, value: torch.Tensor) -> torch.Tensor:
        """Project value tensor for V.

        :param torch.Tensor value: Value tensor.
        :return torch.Tensor: Projected V tensor.
        """
        if self.qkv_same_dim:
            return self._in_proj(value, start=2 * self.embed_dim)
        else:
            weight = self.v_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[2 * self.embed_dim :]
            return F.linear(value, weight, bias)

    def reorder_incremental_state(
        self, incremental_state: Dict[str, Any], new_order: torch.Tensor
    ) -> None:
        """Reorder buffered internal state (for incremental generation).

        :param dict incremental_state: Incremental state buffer.
        :param torch.Tensor new_order: New ordering indices.
        """
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Fetch the incremental state buffer for attention.

        :param dict incremental_state: Incremental state buffer.
        :return Dict[str, torch.Tensor]: Attention state dictionary.
        """
        return (
            utils.get_incremental_state(
                self,
                incremental_state,
                "attn_state",
            )
            or {}
        )

    def _set_input_buffer(
        self, incremental_state: Dict[str, Any], buffer: Dict[str, torch.Tensor]
    ) -> None:
        """Set the incremental state buffer for attention.

        :param dict incremental_state: Incremental state buffer.
        :param Dict[str, torch.Tensor] buffer: Attention state dictionary.
        """
        utils.set_incremental_state(
            self,
            incremental_state,
            "attn_state",
            buffer,
        )

    def apply_sparse_mask(
        self, attn_weights: torch.Tensor, tgt_len: int, src_len: int, bsz: int
    ) -> torch.Tensor:
        """Apply a sparse attention mask (no-op by default).

        :param torch.Tensor attn_weights: Attention weights.
        :param int tgt_len: Target length.
        :param int src_len: Source length.
        :param int bsz: Batch size.
        :return torch.Tensor: Masked attention weights.
        """
        return attn_weights
