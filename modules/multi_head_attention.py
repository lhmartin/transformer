import torch.nn as nn
from math import sqrt
from torch import Tensor
from modules.scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        model_dimension : int = 512,
        key_query_dim  : int = 64,
        value_dim      : int = 64,
        n_heads        : int = 8,
    ) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.embedding_dim = key_query_dim
        
        assert model_dimension % n_heads == 0, "Model Dimension must be divisible by the number of heads"

        self.queries_linear = nn.Linear(model_dimension, key_query_dim)
        self.keys_linear = nn.Linear(model_dimension, key_query_dim)
        self.values_linear = nn.Linear(model_dimension, value_dim)

        self.atn_method = ScaledDotProductAttention()

        self.output_layer = nn.Linear(value_dim, model_dimension)

    def forward(
        self, queries: Tensor, keys: Tensor, values: Tensor, mask=None
    ) -> Tensor:
        queries = self.queries_linear(queries)
        queries = self._reshape_to_heads(queries)

        keys = self.keys_linear(keys)
        keys = self._reshape_to_heads(keys)

        values = self.values_linear(values)
        values = self._reshape_to_heads(values)

        embeddings = self.atn_method(
            queries=queries,
            keys=keys,  # [batch, n_head, seq_len, query_dim]
            values=values,  # [batch, n_head, seq_len, values_dim]
            scaling_factor=sqrt(self.embedding_dim // self.n_heads),  # [1]
            mask=mask,
        )

        # recombine the heads
        embeddings = embeddings.transpose(1, 2).flatten(2, 3)

        embeddings = self.output_layer(embeddings)

        return embeddings

    def _reshape_to_heads(self, input_matrix: Tensor) -> Tensor:
        """Reshape an input matrix into a 4-dim matrix of shape: [batch_size, n_heads, seq_len, embed_size // n_heads]

        Args:
            input_matrix (Tensor): 3-dim matrix of shape : [batch_size, seq_len, embed_size]

        Returns:
            Tensor: The reshaped array of shape : [batch_size, n_heads, seq_len, embed_size // n_heads]
        """
        # extract the shape info
        bs, seq_len, embed_size = input_matrix.shape

        # split the model embedding size into n_heads
        input_matrix = input_matrix.view(
            bs, seq_len, self.n_heads, embed_size // self.n_heads
        )

        # swap the head and sequence length dimensions
        return input_matrix.transpose(1, 2)
