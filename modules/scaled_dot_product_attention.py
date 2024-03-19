from torch import matmul, softmax, Tensor
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        return

    def forward(
        self,
        queries: Tensor,  # [batch x n_head x seq_len x query_dim]
        keys: Tensor,  # [batch x n_head x seq_len x query_dim]
        values: Tensor,  # [batch x n_head x seq_len x values_dim]
        scaling_factor: Tensor,  # [1]
        mask: Tensor | None = None,  # mask out certain input values
    ) -> Tensor:
        # First Q * K -> [batch, n_head, seq_len, seq_len]
        qv = matmul(queries, keys.transpose(-2, -1))

        # Scale
        qv = qv / scaling_factor

        if mask is not None:
            qv = qv.masked_fill(~mask, float('-inf'))

        # Softmax, ie: scale between [0,1] and distribute importance across the input
        atn_values = softmax(qv, dim=-1)

        # Multiple with values, ie: scale values by attention values
        embeddings = matmul(atn_values, values)

        return embeddings
