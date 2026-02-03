"""
Module for defining the Encoder blocks in the transformer
"""

import logging
from typing import List, Optional, Tuple

LOGGER = logging.getLogger(__name__)


import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from annotated_mpnet.constants import position_offset
from annotated_mpnet.transformer_modules import (
    LayerNorm,
    PositionalEmbedding,
    SentenceEncoderLayer,
)


class SentenceEncoder(nn.Module):
    """
    Implementation for a Bi-directional Transformer based Sentence Encoder used
    in BERT/XLM style pre-trained models.
    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    TransformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).
    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens
    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape B x T x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
        self,
        padding_idx: int,
        vocab_size: int,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        max_seq_len: int = 256,
        num_segments: int = 2,
        use_position_embeddings: bool = True,
        offset_positions_by_padding: bool = True,
        encoder_normalize_before: bool = False,
        activation_fn: str = "relu",
        learned_pos_embedding: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        gradient_checkpointing: bool = False,
        relative_attention_num_buckets: int = None,
        relative_attention_max_distance: int = None,
        normalize_before: bool = False,
        export: bool = False,
    ) -> None:
        """
        There is a LOT going on here, so I will try to summarize it all

        Args:
            padding_idx: the index of the padding token
            vocab_size: the total number of tokens in vocab. This will be used to create the
                embedding layer that converts tokens to vectors
            num_encoder_layers: how many SentenceEncoderLayers are in each SentenceEncoder
            embedding_dim: the dimension of the embeddings
            ffn_embedding_dim: the hidden size within the feed-forward network after the
                self-attention calculation
            num_attention_heads: the number of attention heads in each layer of the encoder
            dropout: the dropout prob for non-attention and non-activation layers
            attention_dropout: the dropout prob for the attention mechanism
            activation_dropout: the dropout prob inside the feed-forward network
            max_seq_len: the maximum number of tokens in a sequence. This will determine how large
                the positional embeddings should be
            num_segments: the number of segments within the input tokens. This is akin to BERT-style
                pair encoding where there is a sentence A and a sentence B. MPNet does not use this,
                so you would only want to use this in a BERT-style encoder
            use_position_embeddings: boolean that dictates whether or not positional embeddings
                should be mixed into token embeddings
            offset_positions_by_padding: boolean that dictates whether or not positional embeddings
                should be offset to start at padding_idx + 1. This is usually always True
            encoder_normalize_before: boolean that dictates whether or not a layer norm should be
                applied before or after the attention mechanism in each layer
            activation_fn: the activation function used in the feed-forward network
            learned_pos_embedding: boolean that dictates whether learned positional embeddings or
                sinusoidal positional embeddings should be used
            add_bias_kv: boolean that dictates if a bias parameter should be added to the K and V
                matrices in the attention mechanism
            add_zero_attn: boolean that dictates if zero attention should be added
            embed_scale: a float that will scale all values of the token embeddings before mixing in
                the positional embeddings
            freeze_embeddings: boolean that dictates whether or not the embeddings layers should be
                frozen. This is probably only useful for finetuning
            n_trans_layers_to_freeze: the number of encoder layers to freeze within the encoder.
                This is probably only useful for finetuning
            gradient_checkpointing: whether to enable activation checkpointing for encoder layers
            relative_attention_num_buckets: the number of buckets to add to the relative atttention
                portion of the attention mechanism
            relative_attention_max_distance: the maximum distance (in tokens) to consider in the relative
                attention mechanism
            normalize_before: boolean dictating if a layer norm should be applied before the encoder
                layers
            export: boolean dictating ONNX exporting, which I think we won't be using
        """

        super().__init__()

        # Store all args
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.num_segments = num_segments
        self.use_position_embeddings = use_position_embeddings
        self.learned_pos_embedding = learned_pos_embedding
        self.gradient_checkpointing = gradient_checkpointing

        # Create the embedding layer that will convert token IDs into embeds
        self.embed_tokens = nn.Embedding(self.vocab_size, self.embedding_dim, self.padding_idx)

        # Store more args
        self.embed_scale = embed_scale

        # Get embeddings for token segment. Only created if num_segments > 0
        self.segment_embeddings = (
            nn.Embedding(self.num_segments, self.embedding_dim, padding_idx=None)
            if self.num_segments > 0
            else None
        )

        # Get positonal embeddings
        self.embed_positions = (
            PositionalEmbedding(
                self.max_seq_len,
                self.embedding_dim,
                padding_idx=(self.padding_idx if offset_positions_by_padding else None),
                learned=self.learned_pos_embedding,
            )
            if self.use_position_embeddings
            else None
        )

        # Set up relative attention bias for the attention mechanism
        # and compute params for relative attention if they are not specified
        base_context = 512
        base_buckets = 32  # Default buckets for 512 context length is 32
        base_max_distance = 128  # Default max distance for 512 context length is 128

        if relative_attention_num_buckets is None:
            # linear scaling of num buckets based on seq len (round up to nearest 8)
            scaled_buckets = max(32, int(base_buckets * max_seq_len / base_context))
            self.relative_attention_num_buckets = (scaled_buckets + 7) // 8 * 8
        else:
            self.relative_attention_num_buckets = relative_attention_num_buckets

        if relative_attention_max_distance is None:
            # linear scaling of max distance based on seq len (round up to nearest 8)
            scaled_max_distance = max(128, int(base_max_distance * max_seq_len / base_context))
            self.relative_attention_max_distance = (scaled_max_distance + 7) // 8 * 8
        else:
            self.relative_attention_max_distance = relative_attention_max_distance

        self.relative_attention_bias = nn.Embedding(
            self.relative_attention_num_buckets, num_attention_heads, padding_idx=None
        )

        # Set up the encoder layers in the typical way using a module list
        self.layers = nn.ModuleList(
            [
                SentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    add_bias_kv=add_bias_kv,
                    add_zero_attn=add_zero_attn,
                    normalize_before=normalize_before,
                    export=export,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # Embedding layer norm is independent from pre/post-norm transformer blocks.
        # Embedding vs final norms are distinct by design; checkpoint compatibility is not a
        # requirement for this in-development repo.
        self.emb_layer_norm = (
            LayerNorm(self.embedding_dim, export=export) if encoder_normalize_before else None
        )

        # Final layer norm is only used in pre-norm (normalize_before) configurations.
        self.final_layer_norm = (
            LayerNorm(self.embedding_dim, export=export) if normalize_before else None
        )

        self.normalize_before = normalize_before

        # Define a helper function to freeze embedding layers if specified in the args
        def freeze_module_params(m: nn.Module) -> None:
            """Freeze parameters for a given module.

            :param nn.Module m: Module whose parameters should be frozen.
            """
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        # Now use the helper function to freeze params if specified
        if freeze_embeddings:
            freeze_module_params(self.embed_tokens)
            freeze_module_params(self.segment_embeddings)
            freeze_module_params(self.embed_positions)
            freeze_module_params(self.emb_layer_norm)

        # We can also freeze encoder layers with the n_trans_layers_to_freeze which we process here
        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

    def set_gradient_checkpointing(self, enable: bool = True) -> None:
        """Enable or disable activation checkpointing.

        :param bool enable: Whether to enable checkpointing, defaults to True.
        :return None: This method returns nothing.
        """
        self.gradient_checkpointing = enable

    def _position_offset(self) -> int:
        """Return the position offset implied by the positional embedding padding index.

        :return int: Offset to apply to 0-based positions.
        """
        if self.embed_positions is None:
            return 0
        return position_offset(getattr(self.embed_positions, "padding_idx", None))

    def _position_embeddings_from_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Return positional embeddings for precomputed positions.

        :param torch.Tensor positions: Position indices with shape (bsz, seq_len).
        :return torch.Tensor: Positional embeddings.
        """
        if self.embed_positions is None:
            raise ValueError("Position embeddings are disabled but positions were provided.")

        positions = positions + self._position_offset()

        if hasattr(self.embed_positions, "_forward"):
            return self.embed_positions._forward(positions)

        if hasattr(self.embed_positions, "weight"):
            return F.embedding(positions, self.embed_positions.weight, self.padding_idx)

        raise ValueError("Unsupported positional embedding module for precomputed positions.")

    def encode_emb(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        segment_labels: Optional[torch.Tensor] = None,
        has_padding: Optional[bool] = None,
    ) -> torch.Tensor:
        """Embed input tokens using the encoder's embedding layers.

        :param torch.Tensor input_ids: Input token IDs for the batch.
        :param torch.Tensor positions: Optional precomputed positions for permuted inputs.
        :param torch.Tensor segment_labels: Optional segment labels for token type embeddings.
        :param bool has_padding: Optional CPU-known padding flag to skip padding masks.
        :return torch.Tensor: Embedded token representations.
        """
        padding_mask = None
        if has_padding is not False:
            padding_mask = input_ids.eq(self.padding_idx)

        x = self.embed_tokens(input_ids)
        if self.embed_scale is not None:
            x *= self.embed_scale

        if self.embed_positions is not None:
            if positions is not None:
                x += self._position_embeddings_from_positions(positions)
            else:
                x += self.embed_positions(input_ids)

        if self.segment_embeddings is not None and segment_labels is not None:
            x += self.segment_embeddings(segment_labels)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        if padding_mask is not None:
            x *= 1 - padding_mask.unsqueeze(-1).type_as(x)
        return x

    def maybe_final_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply final layer normalization if configured.

        :param torch.Tensor x: Tensor to normalize.
        :return torch.Tensor: Normalized tensor.
        """
        if self.final_layer_norm is not None:
            return self.final_layer_norm(x)
        return x

    def forward(
        self,
        tokens: torch.Tensor,
        segment_labels: torch.Tensor = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
        has_padding: Optional[bool] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Run the encoder forward pass.

        :param torch.Tensor tokens: Token IDs of shape (bsz, seq_len).
        :param torch.Tensor segment_labels: Segment labels, defaults to None.
        :param bool last_state_only: Whether to return only the final state, defaults to False.
        :param torch.Tensor positions: Precomputed positions, defaults to None.
        :param bool has_padding: Optional CPU-known padding flag to skip padding masks.
        :return Tuple[List[torch.Tensor], torch.Tensor]: Hidden states list and sentence embedding.
        """

        # Compute padding mask. This is needed for multi-head attention.
        padding_mask = None
        if has_padding is not False:
            padding_mask = tokens.eq(self.padding_idx)

        # Get the embeddings for the token sequence
        x = self.embed_tokens(tokens)

        # Scale the embeddings if the appropriate arg is specified
        if self.embed_scale is not None:
            x *= self.embed_scale

        # Add in positional embeddings if they are specified
        if self.embed_positions is not None:
            if positions is not None:
                x += self._position_embeddings_from_positions(positions)
            else:
                x += self.embed_positions(tokens)

        # If there is a segment label, pass those segments into the segment_embedding layer
        if self.segment_embeddings is not None and segment_labels is not None:
            x += self.segment_embeddings(segment_labels)

        # Process the layer norm
        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        # Dropout after the layer norm
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Account for padding while computing the representation
        if padding_mask is not None:
            x *= 1 - padding_mask.unsqueeze(-1).type_as(x)

        # Transpose the batch for easier attention caluclation later on. This is an artifact of the
        # fairseq codebase, but since it's done like this everywhere, we have to keep it
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # Compute the relative attention bias (explicit positions must drive relative bias too).
        if positions is not None:
            if positions.shape != tokens.shape:
                raise ValueError("positions must match tokens shape (bsz, seq_len) when provided.")
            positions_bias = self.compute_position_bias_from_positions(positions.to(torch.long))
        else:
            positions_bias = self.compute_position_bias(
                x, self.relative_attention_num_buckets, self.relative_attention_max_distance
            )
        if positions_bias is not None and (
            positions_bias.device != x.device or positions_bias.dtype != x.dtype
        ):
            positions_bias = positions_bias.to(device=x.device, dtype=x.dtype)

        # If the user wants ALL hidden states, we keep track of it here
        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        # Now process through all the encoder layers (and add each intermediate state if
        # last_state_only is False)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():

                def _layer_forward(
                    layer_input: torch.Tensor, current_layer: nn.Module = layer
                ) -> torch.Tensor:
                    """Run a single encoder layer for checkpointed execution.

                    :param torch.Tensor layer_input: Layer input tensor.
                    :param nn.Module current_layer: Encoder layer module.
                    :return torch.Tensor: Layer output tensor.
                    """
                    output, _ = current_layer(
                        layer_input,
                        self_attn_padding_mask=padding_mask,
                        positions_bias=positions_bias,
                    )
                    return output

                x = checkpoint(_layer_forward, x, use_reentrant=False)
            else:
                x, _ = layer(x, self_attn_padding_mask=padding_mask, positions_bias=positions_bias)
            if not last_state_only:
                inner_states.append(x)

        # Compute the layer norm if the bools evaluate properly
        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        # Transpose the batch back to the standard format
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if not last_state_only:
            # Normalize inner state layout to B x T x C regardless of last_state_only.
            inner_states = [state.transpose(0, 1) for state in inner_states]

        # Get the sentence representation by extracting the CLS token embedding (index 0)
        sentence_rep = x[:, 0, :]

        # If the user only wants the last state only, here's where we add it
        if last_state_only:
            inner_states = [x]

        return inner_states, sentence_rep

    def compute_position_bias(
        self, x: torch.Tensor, num_buckets: int, max_distance: int
    ) -> torch.Tensor:
        """Compute relative position bias for self-attention.

        :param torch.Tensor x: Input tensor with shape (seq_len, batch_size, embed_dim).
        :param int num_buckets: Number of buckets for relative positions.
        :param int max_distance: Maximum relative distance to consider.
        :return torch.Tensor: Relative position bias tensor with shape (heads, qlen, klen).
        """

        # Get q and k len
        qlen, klen = x.size(0), x.size(0)
        device = x.device
        context_position = torch.arange(qlen, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long, device=device)[None, :]

        relative_position = memory_position - context_position

        return self._relative_position_bias(
            relative_position,
            qlen,
            klen,
            num_buckets=num_buckets,
            max_distance=max_distance,
        )

    def compute_position_bias_from_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute relative position bias using explicit position indices.

        :param torch.Tensor positions: Position indices with shape (bsz, seq_len).
        :return torch.Tensor: Relative position bias tensor with shape (bsz * heads, qlen, klen).
        """
        qlen, klen = positions.size(1), positions.size(1)
        context_position = positions[:, :, None]
        memory_position = positions[:, None, :]
        relative_position = memory_position - context_position
        return self._relative_position_bias(relative_position, qlen, klen)

    def _relative_position_bias(
        self,
        relative_position: torch.Tensor,
        qlen: int,
        klen: int,
        num_buckets: Optional[int] = None,
        max_distance: Optional[int] = None,
    ) -> torch.Tensor:
        """Shared helper to bucket relative positions and return attention bias.

        :param torch.Tensor relative_position: Relative position tensor (qlen x klen or bsz x qlen x klen).
        :param int qlen: Query length.
        :param int klen: Key length.
        :param int num_buckets: Optional bucket override.
        :param int max_distance: Optional max distance override.
        :return torch.Tensor: Relative position bias tensor with shape (heads, qlen, klen)
            for shared positions or (bsz * heads, qlen, klen) for per-sample positions.
        """
        if num_buckets is None:
            num_buckets = self.relative_attention_num_buckets
        if max_distance is None:
            max_distance = self.relative_attention_max_distance

        rp_bucket = self.relative_position_bucket(
            relative_position,
            num_buckets=num_buckets,
            max_distance=max_distance,
        )
        rp_bucket = rp_bucket.to(relative_position.device)
        values = self.relative_attention_bias(rp_bucket)

        if relative_position.dim() == 2:
            values = values.permute(2, 0, 1).contiguous()
            # Shared across batch; avoid expanding to (bsz * heads) for sequential positions.
            return values

        values = values.permute(0, 3, 1, 2).contiguous()
        return values.view(-1, qlen, klen)

    @staticmethod
    def relative_position_bucket(
        relative_position: torch.Tensor,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> torch.Tensor:
        """Compute relative position buckets for biasing attention.

        :param torch.Tensor relative_position: Relative positions tensor.
        :param int num_buckets: Number of buckets, defaults to 32.
        :param int max_distance: Maximum distance, defaults to 128.
        :return torch.Tensor: Bucketed relative positions.
        """
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).to(torch.long) * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)

        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        return ret
