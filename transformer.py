from modules.multi_head_attention import MultiHeadAttention
import torch.nn as nn
from torch import Tensor

from modules.positional_encoding import SinCosPositionalEmbedding


class EncoderBlock(nn.Module):
    def __init__(
        self,
        input_features: int = 32,
        n_heads: int = 8,
        embed_dim: int = 64,
        linear_embed: int = 2048,
        droput_prob: float = 0.1,
    ):
        super().__init__()

        self.mh_attention: MultiHeadAttention = MultiHeadAttention(
            input_features=input_features,
            n_heads=n_heads,
            embedding_dim=embed_dim,
        )

        self.position_wise_ff = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=linear_embed),
            nn.ReLU(),
            nn.Linear(in_features=linear_embed, out_features=input_features),
        )

        self.layer_norm = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(p=droput_prob)

    def forward(self, inputs: Tensor):
        """Compute the output of ...forward

        Args:
            inputs (Tensor): Tensor of shape: [batch size, seq_len, embedding_size]

        Returns:
            Tensor:
        """

        embedding = self.mh_attention(queries=inputs, keys=inputs, values=inputs)

        embedding = self.dropout(embedding)

        embedding_normed = self.layer_norm(embedding + inputs)

        final_embedding = self.position_wise_ff(embedding_normed)

        return self.layer_norm(self.dropout(final_embedding) + embedding_normed)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_features: int = 32,
        n_heads: int = 8,
        embed_dim: int = 64,
        linear_embed: int = 2048,
        droput_prob: float = 0.1,
    ):
        super().__init__()

        self.mh_attention: MultiHeadAttention = MultiHeadAttention(
            input_features=input_features,
            n_heads=n_heads,
            embedding_dim=embed_dim,
        )

        self.masked_mh_attention: MultiHeadAttention = MultiHeadAttention(
            input_features=input_features,
            n_heads=n_heads,
            embedding_dim=embed_dim,
        )

        self.position_wise_ff = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=linear_embed),
            nn.ReLU(),
            nn.Linear(in_features=linear_embed, out_features=input_features),
        )

        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.layer_norm_3 = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(p=droput_prob)

    def forward(self, inputs: Tensor, encoder_inputs: Tensor):
        """Compute the output of ...forward

        Args:
            inputs (Tensor): Tensor of shape: [batch size, seq_len, embedding_size]

        Returns:
            Tensor:
        """
        # Sub layer 1
        embedding = self.mh_attention(queries=inputs, keys=inputs, values=inputs)
        embedding = self.dropout(embedding)
        embedding_normed = self.layer_norm_1(embedding + inputs)

        # Sub layer 2
        masked_embedding = self.masked_mh_attention(
            queries=encoder_inputs, keys=encoder_inputs, values=embedding_normed
        )
        embedding_normed = self.layer_norm_2(masked_embedding + embedding_normed)

        # Sub layer 3
        final_embedding = self.position_wise_ff(embedding_normed)
        return self.layer_norm_3(self.dropout(final_embedding) + embedding_normed)


class Transformer(nn.Module):
    def __init__(
        self,
        max_sequence_len: int = 1024,
        token_embedding_dim: int = 10,
        model_dim: int = 64,
        num_encoder_blocks: int = 6,
        num_decoder_blocks: int = 6,
    ):
        self.embedder = nn.Embedding(
            num_embeddings=37000, embedding_dim=token_embedding_dim
        )

        self.pos_encoding = SinCosPositionalEmbedding(
            max_sequence_length=max_sequence_len, model_dim=model_dim
        )

        self.encoder_trunk = nn.Sequential(
            *[
                EncoderBlock(
                    input_features=32,
                    n_heads=8,
                    embed_dim=64,
                    linear_embed=2048,
                    droput_prob=0.1,
                )
            ]
            * num_encoder_blocks
        )

        self.decoder_trunk = nn.Sequential(
            *[
                DecoderBlock(
                    input_features=32,
                    n_heads=8,
                    embed_dim=64,
                    linear_embed=2048,
                    droput_prob=0.1,
                )
            ]
            * num_decoder_blocks
        )

    def forward(self, input_sequence: Tensor) -> Tensor:
        self

        input_with_pos_embeds = self.pos_encoding(input_sequence)


if __name__ == "__main__":
    from torch import randn

    seq_len = 100
    num_heads = 4
    input_dim = 128
    batch_size = 8

    block = EncoderBlock(
        input_features=input_dim,
        n_heads=num_heads,
    )

    x = randn(batch_size, seq_len, input_dim)
    block(x)
