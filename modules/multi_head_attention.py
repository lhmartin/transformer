import torch.nn as nn
from torch import Tensor, sqrt, randn
from modules.scaled_dot_product_attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    
    def __init__(
        self,
        input_features : int = 32,
        n_heads        : int = 8,
        embedding_dim  : int = 64,
        ) -> None:
        super().__init__()
        
        self.input_dim     = input_features
        self.n_heads       = n_heads
        self.embedding_dim = Tensor([embedding_dim])
        
        self.queries_linear = nn.Linear(input_features, embedding_dim)
        self.keys_linear    = nn.Linear(input_features, embedding_dim)
        self.values_linear  = nn.Linear(input_features, embedding_dim)
        
        self.atn_method = ScaledDotProductAttention()
        
        self.output_layer = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, x : Tensor) -> Tensor:

        queries = self.queries_linear(x)
        keys    = self.keys_linear(x)
        values  = self.values_linear(x)
        
        embeddings = self.atn_method(
            queries        = queries,
            keys           = keys,   # [batch * n_head, seq_len, query_dim]
            values         = values, # [batch * n_head, seq_len, values_dim]
            scaling_factor = sqrt(self.embedding_dim), # [1]
        )
        
        embeddings = self.output_layer(embeddings)

        return embeddings

if __name__ == '__main__':
    mha = MultiHeadAttention()
    x = randn(3 * 8, 10, 32)

    print(mha)
    
    mha(x)