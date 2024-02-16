from torch import sin, cos, arange, stack, Tensor

import torch.nn as nn


class SinCosPositionalEmbedding(nn.Module):
    def __init__(self, model_dim: int, max_sequence_length: int):
        super().__init__()
        self.model_dim = model_dim
        self.max_sequence_length = max_sequence_length

        pos_vector = arange(0, max_sequence_length).unsqueeze(0)
        denom = 10000 ** ((2 * arange(0, model_dim)) / model_dim).unsqueeze(0)

        # get only even rows
        sin_embed = sin(pos_vector / denom.T)[0::2]

        # get only odd rows
        cos_embed = cos(pos_vector / denom.T)[1::2]

        # interleave rows
        pos_embedings = stack((sin_embed, cos_embed), dim=1).view(
            model_dim, max_sequence_length
        )

        # make sure the values are not optimized
        self.register_buffer("pos_embedings", pos_embedings)

    def forward(self, inputs: Tensor) -> Tensor:
        # repeat accross the batch dim
        bs, seq_len = inputs.shape[0], inputs.shape[1]
        pos_embeds = self.pos_embedings.unsqueeze(0).repeat(bs, 1, 1).transpose(1,2)

        # add to input
        return inputs + pos_embeds[:,:seq_len, :]


if __name__ == "__main__":
    SinCosPositionalEmbedding(model_dim=16, max_sequence_length=1000)
