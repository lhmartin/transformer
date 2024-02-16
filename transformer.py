from modules.multi_head_attention import MultiHeadAttention
import torch.nn as nn
from torch import Tensor, randint

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

        self.layer_norm = nn.LayerNorm(input_features)

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

        self.layer_norm_1 = nn.LayerNorm(input_features)
        self.layer_norm_2 = nn.LayerNorm(input_features)
        self.layer_norm_3 = nn.LayerNorm(input_features)

        self.dropout = nn.Dropout(p=droput_prob)

    def forward(
        self, inputs: Tensor, encoder_inputs: Tensor, mask: Tensor | None = None
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
        # Sub layer 1
        embedding = self.mh_attention(queries=inputs, keys=inputs, values=inputs)
        embedding = self.dropout(embedding)
        embedding_normed = self.layer_norm_1(embedding + inputs)

        # Sub layer 2
        masked_embedding = self.masked_mh_attention(
            queries=encoder_inputs,
            keys=encoder_inputs,
            values=embedding_normed,
            mask=mask,
        )

        embedding_normed = self.layer_norm_2(masked_embedding + embedding_normed)

        # Sub layer 3
        final_embedding = self.position_wise_ff(embedding_normed)
        return self.layer_norm_3(self.dropout(final_embedding) + embedding_normed)


class Transformer(nn.Module):
    def __init__(
        self,
        max_sequence_len: int = 1024,
        model_dim: int = 64,
        num_encoder_blocks: int = 6,
        num_decoder_blocks: int = 6,
    ):
        super().__init__()
        self.source_embedder = nn.Embedding(
            # TODO: need to figure out how many tokens
            num_embeddings=37000,
            embedding_dim=model_dim,
        )
        self.decoder_embedder = nn.Embedding(
            # TODO: need to figure out how many tokens
            num_embeddings=37000,
            embedding_dim=model_dim,
        )

        self.pos_encoding = SinCosPositionalEmbedding(
            max_sequence_length=max_sequence_len, model_dim=model_dim
        )

        self.encoder_trunk = nn.ModuleList(
            [
                EncoderBlock(
                    input_features=model_dim,
                    n_heads=8,
                    embed_dim=64,
                    linear_embed=2048,
                    droput_prob=0.1,
                )
            ]
            * num_encoder_blocks
        )

        self.decoder_trunk = nn.ModuleList(
            [
                DecoderBlock(
                    input_features=model_dim,
                    n_heads=8,
                    embed_dim=64,
                    linear_embed=2048,
                    droput_prob=0.1,
                )
            ]
            * num_decoder_blocks
        )
        
        self.final_linear = nn.Linear(model_dim, 37000)
        self.softmax = nn.Softmax(2)

    def encode(self, input_sequence: Tensor) -> Tensor:
        tokens = self.source_embedder(input_sequence)

        # add positional embeddings
        tokens = self.pos_encoding(tokens)
        
        for encode_block in self.encoder_trunk:
            tokens = encode_block(tokens)

        return tokens

    def decode(self, input_embeddings: Tensor, target: Tensor) -> Tensor:
        decode_tkns = self.decoder_embedder(target)

        mask = self.make_mask(input_embeddings, decode_tkns)
        
        for decode_block in self.decoder_trunk:
            decode_embeddings = decode_block(input_embeddings, decode_tkns, mask)

        logits = self.final_linear(decode_embeddings)

        return self.softmax(logits)

    def make_mask(self, input_embeddings, target_embeddings) -> Tensor:
        return None
    
    def forward(self, input_embeddings : Tensor, target_embeddings : Tensor):

        input_embeddings = self.encode(input_embeddings)

        return self.decode(input_embeddings, target_embeddings)
        
        


if __name__ == "__main__":
    
    model = Transformer()
    src_vocab_size = 2000
    
    random_input   = randint(1, src_vocab_size, (64, 100))
    # random_decode  = randint(1, src_vocab_size, (64, 100))
    
    model.forward(random_input, random_input)
    
