from torch import matmul, softmax, Tensor
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
                ) -> Tensor:
         
        # First Q * K -> [batch x n_head x seq_len x query_dim]
        qv = matmul(queries, keys.permute(0, 2, 1))
        
        # Scale
        scaled = qv /scaling_factor
        
        # Softmax, ie: scale between [0,1] and distribute importance across the input
        atn_values = softmax(scaled, dim=2)

        # Multiple with values, ie: scale values by attention values
        embeddings  = matmul(atn_values, values)
        
        return embeddings