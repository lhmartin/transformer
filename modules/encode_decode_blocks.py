import torch.nn as nn
from torch import Tensor
from modules.multi_head_attention import MultiHeadAttention

class EncoderBlock(nn.Module):
    def __init__(
        self,
        model_dimension : int = 512,
        n_heads         : int = 8,
        value_dim       : int = 64,
        key_query_dim   : int = 64,
        ff_embed        : int = 2048,
        droput_prob     : float = 0.1
    ):
        super().__init__()

        self.mh_attention: MultiHeadAttention = MultiHeadAttention(
            model_dimension = model_dimension,
            key_query_dim = key_query_dim,
            value_dim = value_dim,
            n_heads = n_heads
        )

        self.position_wise_ff = nn.Sequential(
            nn.Linear(in_features=model_dimension, out_features=ff_embed),
            nn.ReLU(),
            nn.Linear(in_features=ff_embed, out_features=model_dimension),
        )

        self.layer_norm = nn.LayerNorm(model_dimension)

        self.dropout = nn.Dropout(p=droput_prob)

    def forward(self, inputs: Tensor, mask : Tensor | None = None):
        """Encode using self-attention.

        Args:
            inputs (Tensor): Tensor of shape: [batch_size, seq_len, embedding_size]
            mask (Tensor | None):
                Tensor of booleans with shape: [batch size, 1, 1, seq_len].
                If mask is present, stops the encoder from attending to any padding tokens.
                If None, no masking is done, and the model can attend to all input tokens.

        Returns:
            Tensor: A tensor containing embeddings of the input, with shape [batch_size, seq_len, model_dimension]
        """
        embedding = self.mh_attention(queries=inputs, keys=inputs, values=inputs, mask=mask)

        embedding = self.dropout(embedding)

        embedding_normed = self.layer_norm(embedding + inputs)

        final_embedding = self.position_wise_ff(embedding_normed)

        return self.layer_norm(self.dropout(final_embedding) + embedding_normed)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        model_dimension : int = 512,
        n_heads         : int = 8,
        value_dim       : int = 64,
        key_query_dim   : int = 64,
        ff_embed        : int = 2048,
        droput_prob     : float = 0.1,
    ):
        super().__init__()

        self.mh_attention: MultiHeadAttention = MultiHeadAttention(
            model_dimension = model_dimension,
            key_query_dim = key_query_dim,
            value_dim = value_dim,
            n_heads = n_heads
        )

        self.masked_mh_attention: MultiHeadAttention = MultiHeadAttention(
            model_dimension = model_dimension,
            key_query_dim = key_query_dim,
            value_dim = value_dim,
            n_heads = n_heads
        )

        self.position_wise_ff = nn.Sequential(
            nn.Linear(in_features=model_dimension, out_features=ff_embed),
            nn.ReLU(),
            nn.Linear(in_features=ff_embed, out_features=model_dimension),
        )

        self.layer_norm_1 = nn.LayerNorm(model_dimension)
        self.layer_norm_2 = nn.LayerNorm(model_dimension)
        self.layer_norm_3 = nn.LayerNorm(model_dimension)

        self.dropout = nn.Dropout(p=droput_prob)

    def forward(
        self, target_embeddings: Tensor, input_embeddings: Tensor, src_mask: Tensor | None = None, trg_mask: Tensor | None = None
    ):
        """Compute the

        Args:
            inputs (Tensor): Tensor of shape: [batch size, seq_len, embedding_size]
                The embedded tokens of the output
            encoder (Tensor): Tensor of shape: [batch size, seq_len, embedding_size]
                The output of the encoder block

        Returns:
            Tensor:
        """
        # Sub layer target_embeddings
        embedding = self.mh_attention(queries=target_embeddings, keys=target_embeddings, values=target_embeddings, mask=trg_mask)
        embedding = self.dropout(embedding)
        embedding_normed = self.layer_norm_1(embedding + target_embeddings)

        # Sub layer 2
        masked_embedding = self.masked_mh_attention(
            queries=embedding_normed,
            keys=input_embeddings,
            values=input_embeddings,
            mask=src_mask,
        )

        embedding_normed = self.layer_norm_2(masked_embedding + embedding_normed)

        # Sub layer 3
        final_embedding = self.position_wise_ff(embedding_normed)
        return self.layer_norm_3(self.dropout(final_embedding) + embedding_normed)
