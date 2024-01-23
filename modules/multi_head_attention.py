import torch.nn as nn
from torch import Tensor, sqrt, matmul, softmax, sum, randn


class MultiHeadAttention(nn.Module):
    
    def __init__(
        self,
        input_dim     : int = 32,
        n_heads       : int = 8,
        embedding_dim : int = 64,
        ) -> None:
        super().__init__()
        
        self.input_dim     = input_dim
        self.n_heads       = n_heads
        self.embedding_dim = Tensor(embedding_dim)

        self.W_q = nn.Parameter(randn(1, n_heads, input_dim, embedding_dim), requires_grad=True)
        self.W_k = nn.Parameter(randn(1, n_heads, input_dim, embedding_dim), requires_grad=True)
        self.W_v = nn.Parameter(randn(1, n_heads, input_dim, embedding_dim), requires_grad=True)
        
        self.mlp_layer = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, x : Tensor) -> Tensor:
        
        queries = x * self.W_q
        keys    = x * self.W_k
        values  = x * self.W_v
        
        atn_values = softmax(matmul(queries, keys) / 
                             sqrt(self.embedding_dim), 
                             dim=2)

        embeddings  = sum(matmul(atn_values, values), dim=2)
        
        embeddings = self.mlp_layer(embeddings)

        return embeddings

if __name__ == '__main__':
    mha = MultiHeadAttention()
    x = randn(10, 10, 32)

    print(mha)
    
    mha(x)