from torch import sin, cos, arange, zeros, Tensor, pow

import torch.nn as nn


class SinCosPositionalEmbedding(nn.Module):
    def __init__(self, model_dimension: int, max_sequence_length: int):
        super().__init__()
        self.model_dimension = model_dimension
        self.max_sequence_length = max_sequence_length

        pos_vector = arange(0, max_sequence_length).unsqueeze(1)
        denom = pow(10000.0, -arange(0,model_dimension,2)/model_dimension)

        # get only even rows
        sin_embed = sin(pos_vector * denom)

        # get only odd rows
        cos_embed = cos(pos_vector * denom)

        pos_embedings = zeros(max_sequence_length, model_dimension)
        pos_embedings[:, 0::2] = sin_embed
        pos_embedings[:, 1::2] = cos_embed

        # make sure the values are not optimized
        self.register_buffer("pos_embedings", pos_embedings)

    def forward(self, inputs: Tensor) -> Tensor:
        # repeat accross the batch dim
        seq_len = inputs.shape[1]
        pos_embeds = self.pos_embedings.unsqueeze(0).transpose(1,2)

        # add to input
        return inputs + pos_embeds[:,:seq_len, :]


if __name__ == "__main__":
    SinCosPositionalEmbedding(model_dimension=16, max_sequence_length=1000)
