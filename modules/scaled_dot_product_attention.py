from torch import matmul, softmax, Tensor, inf
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        return
    
    def forward(self,
                queries        : Tensor, # [batch x n_head x seq_len x query_dim]
                keys           : Tensor, # [batch x n_head x seq_len x query_dim]
                values         : Tensor, # [batch x n_head x seq_len x values_dim]
                scaling_factor : Tensor, # [1]
                mask           : Tensor | None = None # mask out certain input values
                ) -> Tensor:
         
        # First Q * K -> [batch, n_head, seq_len, seq_len]
        qv = matmul(queries, keys.permute(0, 1, 3, 2))

        # Scale
        scaled = qv /scaling_factor

        if mask:
            scaled[mask] = -float('inf')

        # Softmax, ie: scale between [0,1] and distribute importance across the input
        atn_values = softmax(scaled, dim=2)

        # Multiple with values, ie: scale values by attention values
        embeddings  = matmul(atn_values, values)
        
        return embeddings