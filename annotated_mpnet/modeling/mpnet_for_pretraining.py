"""
Module containing the necessary classes for MPNet pretraining, ported directly from fairseq research
code
"""

import logging
from typing import Any, Optional, Sequence, Tuple, Union

from rich.logging import RichHandler

LOG_FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=LOG_FORMAT, datefmt="[%X] ", handlers=[RichHandler()])
LOGGER = logging.getLogger(__name__)


import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedTokenizer

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
        if module.padding_idx:
            module.weight.data[module.padding_idx].zero_()


class MPNetForPretraining(nn.Module):
    """
    Class containing all the methods required for pretraining MPNet
    """

    def __init__(self, args: Any, tokenizer: PreTrainedTokenizer) -> None:
        """Initialize the MPNet pretraining model.

        :param object args: Configuration namespace containing model hyperparameters.
        :param PreTrainedTokenizer tokenizer: Tokenizer used to determine vocab and padding IDs.
        """
        super().__init__()

        # Use padded_vocab_size if available, otherwise use the tokenizer's vocab_size
        vocab_size = getattr(args, "padded_vocab_size", tokenizer.vocab_size)

        # Let's define the encoder here
        self.args = args
        self.sentence_encoder = SentenceEncoder(
            padding_idx=tokenizer.vocab[tokenizer.pad_token],
            vocab_size=vocab_size,  # Use the padded vocab size
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
            relative_attention_num_buckets=args.relative_attention_num_buckets,
            relative_attention_max_distance=args.relative_attention_max_distance,
        )

        # Add the language modeling head
        self.lm_head = MPNetLMHead(
            embed_dim=args.encoder_embed_dim,
            output_dim=vocab_size,  # Use the padded vocab size
            activation_fn=args.activation_fn,
            weight=self.sentence_encoder.embed_tokens.weight,
        )

        # Initialize the weights
        self.apply(init_final_params)

    def output_layer(
        self, features: torch.Tensor, masked_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Project encoder features to vocabulary logits.

        :param torch.Tensor features: Encoder features.
        :param torch.Tensor masked_tokens: Mask positions to select, defaults to None.
        :return torch.Tensor: Vocabulary logits.
        """
        return self.lm_head(features, masked_tokens)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        pred_size: int,
        return_mlm: bool = False,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute the MPNet forward pass.

        :param torch.Tensor input_ids: Input token IDs.
        :param torch.Tensor positions: Position indices for the batch.
        :param int pred_size: Number of tokens to predict.
        :param bool return_mlm: Whether to return an additional MLM head output, defaults to False.
        :param dict kwargs: Additional unused keyword arguments.
        :return torch.Tensor: Vocabulary logits (or logits tuple when ``return_mlm`` is True).
        """

        # Calculate initial embeddings
        emb = self.encode_emb(self.sentence_encoder, input_ids, positions)

        # Reverse the tensor for easier extraction
        x = reverse_tensor(emb)

        # Separate out content and query streams
        c, q = split_tensor(x, pred_size)

        # Get the content and query position biases
        content_position_bias = self.encode_relative_emb(
            self.sentence_encoder, positions[:, :-pred_size]
        )
        query_position_bias = content_position_bias[:, -pred_size:].contiguous()

        # Get the sz of the inital src_length without the tokens to be predicted
        sz = c.size(0) - pred_size

        # Get the query and content masks using the helper function below
        query_mask, content_mask = make_query_and_content_mask(input_ids, sz, pred_size)

        # Do the attention calculations
        for i, layer in enumerate(self.sentence_encoder.layers):
            c, q = encode_two_stream_attention(
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
        q = reverse_tensor(q)

        # Project the attention features out to the vocab size for masked token classification
        x = self.output_layer(q)

        # If we also want MLM loss, we can branch to the below logic. Probably not useful for us
        if return_mlm is True:
            c = c[-pred_size:]
            c = self.maybe_final_norm(self.decoder.sentence_encoder, c)
            c = reverse_tensor(c)
            c = self.output_layer(c)
            return x, c

        return x

    # We define some class static methods here that will be used quite a bit across the board
    @staticmethod
    def encode_emb(
        self, input_ids: torch.Tensor, positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Embed input tokens using the sentence encoder.

        :param SentenceEncoder self: Sentence encoder instance providing embedding layers.
        :param torch.Tensor input_ids: Input token IDs for the batch.
        :param torch.Tensor positions: Precomputed position indices, defaults to None.
        :return torch.Tensor: Embedded token representations.
        """

        # Use the embedding layer of the sentence encoder to embed these (passed in via the self
        # arg)
        x = self.embed_tokens(input_ids)

        # Scale the embeddings if necessary
        if self.embed_scale is not None:
            x *= self.embed_scale

        # Add in positions
        if positions is not None:
            x += F.embedding(positions + 2, self.embed_positions.weight, self.padding_idx)

        # Do layer norm
        if self.emb_layer_norm is not None and not self.normalize_before:
            x = self.emb_layer_norm(x)

        # Process dropout
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    @staticmethod
    def maybe_final_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply final layer normalization if configured.

        :param SentenceEncoder self: Sentence encoder instance.
        :param torch.Tensor x: Tensor to normalize.
        :return torch.Tensor: Normalized tensor.
        """
        if self.emb_layer_norm is not None and self.normalize_before:
            return self.emb_layer_norm(x)
        return x

    @staticmethod
    def encode_relative_emb(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute relative position bias embeddings.

        :param SentenceEncoder self: Sentence encoder instance.
        :param torch.Tensor positions: Position indices for the batch.
        :return torch.Tensor: Relative position bias values.
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
        self,
        embed_dim: int,
        output_dim: int,
        activation_fn: str,
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize the LM head.

        :param int embed_dim: Encoder embedding dimension (typically 768).
        :param int output_dim: Output vocabulary dimension.
        :param str activation_fn: Activation function name for the head.
        :param torch.Tensor weight: Optional shared embedding weights, defaults to None.
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

    def forward(
        self, features: torch.Tensor, masked_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for the LM head.

        :param torch.Tensor features: Encoder outputs.
        :param torch.Tensor masked_tokens: Mask positions to select, defaults to None.
        :return torch.Tensor: Vocabulary logits.
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


# Helper functions below!
def reverse_tensor(x: torch.Tensor) -> torch.Tensor:
    """Reverse the time and batch dimensions of a tensor.

    :param torch.Tensor x: Input tensor.
    :return torch.Tensor: Transposed tensor.
    """
    return x.transpose(0, 1)


def split_tensor(x: torch.Tensor, split_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a tensor into content and query streams.

    :param torch.Tensor x: Tensor to split.
    :param int split_size: Prediction size (query length).
    :return Tuple[torch.Tensor, torch.Tensor]: Content and query tensors.
    """
    # Get the content stream size by subtracting out the pred_size aka split_size
    sz = x.size(0) - split_size

    return x[:sz].contiguous(), x[sz:].contiguous()


def encode_two_stream_attention(
    self: Any,
    c: torch.Tensor,
    q: torch.Tensor,
    content_mask: Optional[torch.Tensor] = None,
    query_mask: Optional[torch.Tensor] = None,
    content_position_bias: Optional[torch.Tensor] = None,
    query_position_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute two-stream attention for a single encoder layer.

    :param object self: Encoder layer instance.
    :param torch.Tensor c: Content stream tensor.
    :param torch.Tensor q: Query stream tensor.
    :param torch.Tensor content_mask: Content attention mask, defaults to None.
    :param torch.Tensor query_mask: Query attention mask, defaults to None.
    :param torch.Tensor content_position_bias: Content position bias, defaults to None.
    :param torch.Tensor query_position_bias: Query position bias, defaults to None.
    :return Tuple[torch.Tensor, torch.Tensor]: Updated content and query tensors.
    """

    def skip_norm_ff_fn(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Apply skip connection, normalization, and feed-forward block.

        :param torch.Tensor x: Input tensor.
        :param torch.Tensor residual: Residual tensor for skip connection.
        :return torch.Tensor: Processed tensor.
        """

        # Calculate dropout
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Process skip connection
        x = x + residual

        # Do normalization where appropriate based on the normalize_before param
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        # Save x as residual for the skip connection AFTER the feed-forard net
        residual = x

        # Normalize again
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)

        # Process the feed-forward net connections with specified activation function and dropout
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Process the skip connection after running through the FF net
        x = x + residual

        # Do a layer norm based on args
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)

        return x

    # Save c and q as residuals for skip connection after attention calculation
    residual_c = c
    residual_q = q

    # Do a normalization before if the class args allow for it
    c = self.maybe_layer_norm(self.self_attn_layer_norm, c, before=True)
    q = self.maybe_layer_norm(self.self_attn_layer_norm, q, before=True)

    # Wrapper function on top of each layer's self attention mechanism that calculates the proper
    # two stream attention that is required for MPNet
    c, q = two_stream_self_attention(
        self.self_attn,
        query=[c, q],
        key=c,
        value=c,
        query_mask=query_mask,
        content_mask=content_mask,
        query_position_bias=query_position_bias,
        content_position_bias=content_position_bias,
    )

    # Calculate skip connection, inner layer norms, and feed forward after attention calculation
    # using the resuable function we built above
    c = skip_norm_ff_fn(c, residual_c)
    q = skip_norm_ff_fn(q, residual_q)

    # Finally return the tensors after the full layer calculation
    return c, q


def two_stream_self_attention(
    self: Any,
    query: Sequence[torch.Tensor],
    key: Optional[torch.Tensor] = None,
    value: Optional[torch.Tensor] = None,
    query_mask: Optional[torch.Tensor] = None,
    content_mask: Optional[torch.Tensor] = None,
    query_position_bias: Optional[torch.Tensor] = None,
    content_position_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute two-stream self-attention using encoder attention weights.

    :param object self: Encoder attention instance.
    :param Sequence[torch.Tensor] query: Content and query tensors.
    :param torch.Tensor key: Key tensor, defaults to None.
    :param torch.Tensor value: Value tensor, defaults to None.
    :param torch.Tensor query_mask: Query attention mask, defaults to None.
    :param torch.Tensor content_mask: Content attention mask, defaults to None.
    :param torch.Tensor query_position_bias: Query position bias, defaults to None.
    :param torch.Tensor content_position_bias: Content position bias, defaults to None.
    :return Tuple[torch.Tensor, torch.Tensor]: Updated content and query tensors.
    """

    # Unpack the content and query tensors from the (poorly) named query arg
    c, q = query

    # Get dimensions
    bsz, embed_dim = key.size(1), key.size(2)

    # Define a few in-scope helper functions that we will be reusing a bunch
    def transpose_fn(x: torch.Tensor) -> torch.Tensor:
        """Transpose to (batch*heads, seq_len, head_dim) for attention.

        :param torch.Tensor x: Input tensor.
        :return torch.Tensor: Transposed tensor.
        """
        return x.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def fill_mask(attn_weights: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Apply attention mask to attention weights.

        :param torch.Tensor attn_weights: Attention weights tensor.
        :param torch.Tensor attn_mask: Mask tensor.
        :return torch.Tensor: Masked attention weights.
        """
        return attn_weights.masked_fill(attn_mask.unsqueeze(0), float("-inf"))

    def attn_fn(
        _q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention for either content or query stream.

        :param torch.Tensor _q: Query tensor.
        :param torch.Tensor k: Key tensor.
        :param torch.Tensor v: Value tensor.
        :param torch.Tensor mask: Attention mask, defaults to None.
        :param torch.Tensor bias: Position bias, defaults to None.
        :return torch.Tensor: Attention output.
        """
        # Process the query matrix through both the scaling and the input layer of self_attention
        _q = transpose_fn(self.scaling * self.in_proj_q(_q))

        # Calculate the energy by multiplying Q and K
        attn_weights = torch.bmm(_q, k.transpose(1, 2))

        # Process bias if applicable
        if bias is not None:
            attn_weights += bias

        # Process attention masking
        if mask is not None:
            attn_weights = fill_mask(attn_weights, mask)

        # Softmax the energy to get the final attention weights
        attn_weights = F.softmax(attn_weights, dim=-1).type_as(attn_weights)

        # Do the attention dropout
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Combine the final post-softmax/dropout weights with the V matrix to get the attention
        attn = torch.bmm(attn_weights, v)

        # Finally, transpose back to the embed dimension and return
        attn = attn.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)

        return self.out_proj(attn)

    # Get K and V matrices by processing them through the input layer for each matrix and transpose
    # them to be the right shape
    k = transpose_fn(self.in_proj_k(key))
    v = transpose_fn(self.in_proj_v(value))

    # Calculate query attention and content attention using the function above
    c = attn_fn(c, k, v, mask=content_mask, bias=content_position_bias)
    q = attn_fn(q, k, v, mask=query_mask, bias=query_position_bias)

    return c, q


def make_query_and_content_mask(
    input_ids: torch.Tensor, seq_len: int, pred_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create content and query masks for MPNet two-stream attention.

    :param torch.Tensor input_ids: Input IDs for device placement.
    :param int seq_len: Sequence length of the input.
    :param int pred_size: Size of the predicted subsequence.
    :return Tuple[torch.Tensor, torch.Tensor]: Query and content attention masks.

    It looks like the below with comparisons to how it's different than XLNet-style PLM:
        Query Mask:
        | <-   PLM   -> |    | <-      MPNet    -> |
        [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 1 1 0 0 0 ]
        [ 0 0 0 0 0 1 1 ]    [ 0 0 0 0 0 1 1 1 0 0 ]
        [ 0 0 0 0 0 0 1 ]    [ 0 0 0 0 0 0 1 1 1 0 ]
        Content Mask:
        | <-   PLM   -> |    | <-      MPNet    -> |
                               x x x x x x x m m m
                               1 2 3 4 5 6 7 5 6 7
        [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 1 1 0 0 0 ]
        [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 1 1 0 0 0 ]
        [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 1 1 0 0 0 ]
        [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 1 1 0 0 0 ]
        [ 0 0 0 0 0 1 1 ]    [ 0 0 0 0 0 1 1 1 0 0 ]
        [ 0 0 0 0 0 0 1 ]    [ 0 0 0 0 0 0 1 1 1 0 ]
        [ 0 0 0 0 0 0 0 ]    [ 0 0 0 0 0 0 0 1 1 1 ]
                             [ 0 0 0 0 1 1 1 0 0 0 ]
                             [ 0 0 0 0 1 1 1 0 0 0 ]
                             [ 0 0 0 0 1 1 1 0 0 0 ]

    Note: This function is designed to scale automatically with sequence length as it's
    matrix-based and constructs masks based on the provided seq_len and pred_size.
    There's no need to modify this function when changing context length.
    """

    # Define helper function to keep things organized
    def make_query_mask() -> torch.Tensor:
        """Build the query attention mask.

        :return torch.Tensor: Query mask tensor.
        """
        # Create the mask portion (i.e. ones)
        mask = torch.triu(torch.ones(pred_size, pred_size), 0)

        mask = (torch.ones(pred_size, seq_len - pred_size), 1 - mask, mask)

        return torch.cat(mask, dim=-1).eq(0)

    def make_content_mask() -> torch.Tensor:
        """Build the content attention mask.

        :return torch.Tensor: Content mask tensor.
        """
        mask = [
            torch.zeros(seq_len - pred_size, pred_size),
            torch.tril(torch.ones(pred_size, pred_size), 0),
        ]

        mask.append(torch.zeros(pred_size, pred_size))
        mask = torch.cat(mask, dim=0)
        mask = (torch.ones(seq_len + pred_size, seq_len - pred_size), mask, 1 - mask)

        return torch.cat(mask, dim=-1).eq(0)

    return make_query_mask().to(input_ids.device), make_content_mask().to(input_ids.device)
