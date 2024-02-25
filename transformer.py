from typing import Dict, Tuple
from modules.multi_head_attention import MultiHeadAttention
import torch.nn as nn
from torch import Tensor, randint, triu, ones

from modules.positional_encoding import SinCosPositionalEmbedding
from pydantic import BaseModel
from transformers import AutoTokenizer
from math import sqrt



class EncoderBlock(nn.Module):
    def __init__(
        self,
        model_dimension: int = 512,
        n_heads: int = 8,
        value_dim: int = 64,
        key_query_dim: int = 64,
        ff_embed: int = 2048,
        droput_prob: float = 0.1,
    ):
        super().__init__()

        self.mh_attention: MultiHeadAttention = MultiHeadAttention(
            model_dimension = model_dimension,
            key_query_dim = key_query_dim,
            value_dim = value_dim,
            n_heads = n_heads
        )

        self.position_wise_ff = nn.Sequential(
            nn.Linear(in_features=model_dimension, out_features=ff_embed),
            nn.ReLU(),
            nn.Linear(in_features=ff_embed, out_features=model_dimension),
        )

        self.layer_norm = nn.LayerNorm(model_dimension)

        self.dropout = nn.Dropout(p=droput_prob)

    def forward(self, inputs: Tensor, mask : Tensor | None = None):
        """Compute the output of ...forward

        Args:
            inputs (Tensor): Tensor of shape: [batch size, seq_len, embedding_size]

        Returns:
            Tensor:
        """

        embedding = self.mh_attention(queries=inputs, keys=inputs, values=inputs, mask=mask)

        embedding = self.dropout(embedding)

        embedding_normed = self.layer_norm(embedding + inputs)

        final_embedding = self.position_wise_ff(embedding_normed)

        return self.layer_norm(self.dropout(final_embedding) + embedding_normed)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        model_dimension: int = 512,
        n_heads: int = 8,
        value_dim: int = 64,
        key_query_dim: int = 64,
        ff_embed: int = 2048,
        droput_prob: float = 0.1,
    ):
        super().__init__()

        self.mh_attention: MultiHeadAttention = MultiHeadAttention(
            model_dimension = model_dimension,
            key_query_dim = key_query_dim,
            value_dim = value_dim,
            n_heads = n_heads
        )

        self.masked_mh_attention: MultiHeadAttention = MultiHeadAttention(
            model_dimension = model_dimension,
            key_query_dim = key_query_dim,
            value_dim = value_dim,
            n_heads = n_heads
        )

        self.position_wise_ff = nn.Sequential(
            nn.Linear(in_features=model_dimension, out_features=ff_embed),
            nn.ReLU(),
            nn.Linear(in_features=ff_embed, out_features=model_dimension),
        )

        self.layer_norm_1 = nn.LayerNorm(model_dimension)
        self.layer_norm_2 = nn.LayerNorm(model_dimension)
        self.layer_norm_3 = nn.LayerNorm(model_dimension)

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
            queries=embedding_normed,
            keys=encoder_inputs,
            values=encoder_inputs,
            mask=mask,
        )

        embedding_normed = self.layer_norm_2(masked_embedding + embedding_normed)

        # Sub layer 3
        final_embedding = self.position_wise_ff(embedding_normed)
        return self.layer_norm_3(self.dropout(final_embedding) + embedding_normed)


class Transformer(nn.Module):

    class Config(BaseModel):
        max_sequence_len   : int = 512
        src_vocab_size     : int = 37000
        tgt_vocab_size     : int = 37000
        num_encoder_blocks : int = 6
        num_decoder_blocks : int = 6
        num_heads          : int = 8
        model_dimension    : int = 512
        value_dim          : int = 512
        key_query_dim      : int = 512
        ff_dim             : int = 2048
        dropout_prob       : float = 0.1

    def __init__(self, config : Config):
        super().__init__()

        self._config = config

        self.source_embedder = nn.Embedding(
            # TODO: need to figure out how many tokens
            num_embeddings=self._config.src_vocab_size,
            embedding_dim=self._config.model_dimension,
        )
        self.decoder_embedder = nn.Embedding(
            # TODO: need to figure out how many tokens
            num_embeddings=self._config.tgt_vocab_size,
            embedding_dim=self._config.model_dimension,
        )

        self.pos_encoding_enc = SinCosPositionalEmbedding(
            max_sequence_length=self._config.max_sequence_len, 
            model_dimension=self._config.model_dimension
        )
        self.pos_encoding_dec = SinCosPositionalEmbedding(
            max_sequence_length=self._config.max_sequence_len, 
            model_dimension=self._config.model_dimension
        )

        self.encoder_trunk = nn.ModuleList(
            [
                EncoderBlock(
                    model_dimension = self._config.model_dimension,
                    n_heads = self._config.num_heads,
                    value_dim = self._config.value_dim,
                    key_query_dim = self._config.key_query_dim,
                    ff_embed = self._config.ff_dim,
                    droput_prob = self._config.dropout_prob,
                )
            ]
            * self._config.num_encoder_blocks
        )

        self.decoder_trunk = nn.ModuleList(
            [
                DecoderBlock(
                    model_dimension = self._config.model_dimension,
                    n_heads = self._config.num_heads,
                    value_dim = self._config.value_dim,
                    key_query_dim = self._config.key_query_dim,
                    ff_embed = self._config.ff_dim,
                    droput_prob = self._config.dropout_prob,
                )
            ]
            * self._config.num_decoder_blocks
        )

        self.final_linear = nn.Linear(self._config.model_dimension, self._config.tgt_vocab_size)
        self.softmax = nn.LogSoftmax(-1)
        self.dropout = nn.Dropout(self._config.dropout_prob)

        self.tokenizer_en = AutoTokenizer.from_pretrained('bert-base-cased')
        self.tokenizer_de = AutoTokenizer.from_pretrained('bert-base-german-cased')

    def encode(self, input_sequence: Tensor, mask : Tensor | None = None) -> Tensor:
        
        tokens = self.source_embedder(input_sequence) * sqrt(self._config.model_dimension)

        # add positional embeddings
        tokens = self.pos_encoding_enc(tokens)
        tokens = self.dropout(tokens)

        for encode_block in self.encoder_trunk:
            tokens = encode_block(tokens, mask)

        return tokens

    def decode(self,
               input_embeddings: Tensor,
               target: Tensor,
               mask : Tensor | None = None) -> Tensor:

        decode_tkns = self.decoder_embedder(target)
        decode_tkns = self.pos_encoding_dec(decode_tkns)
        decode_tkns = self.dropout(decode_tkns)

        for decode_block in self.decoder_trunk:
            decode_embeddings = decode_block(input_embeddings, decode_tkns, mask)

        logits = self.final_linear(decode_embeddings)

        return self.softmax(logits)

    def make_mask(self, input_tkns : Tensor, target_tkns : Tensor) -> Tuple[Tensor, Tensor]:

        seq_len         = input_tkns.size(1)

        input_padding_mask = (input_tkns != self.tokenizer_en.pad_token_id).unsqueeze(-1)
        trgt_padding_mask = (target_tkns != self.tokenizer_de.pad_token_id).unsqueeze(-1)

        look_ahead_mask = (1 - triu(ones(1, seq_len,seq_len), diagonal=1)).bool().to(self.device)

        # combine
        src_mask = (look_ahead_mask & input_padding_mask).unsqueeze(1)
        trgt_mask = (look_ahead_mask & trgt_padding_mask).unsqueeze(1)

        return src_mask, trgt_mask

    def forward(self, input_tkns : Tensor, target_tkns : Tensor):

        src_mask, trgt_mask = self.make_mask(input_tkns, target_tkns)

        input_embeddings = self.encode(input_tkns, mask=src_mask)

        return self.decode(input_embeddings=input_embeddings, target=target_tkns, mask=trgt_mask)

    def init_params(self, default_initialization=False):
        # Not mentioned in the paper, but other implementations used xavier.
        # I tested both PyTorch's default initialization and this, and xavier has tremendous impact! I didn't expect
        # a model's perf, with normalization layers, to be so much dependent on the choice of weight initialization.
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    @property
    def device(self):
        return next(self.parameters()).device

    def collate_fn(self, inputs : Dict[str, str]) -> Dict[str, Dict[str, Tensor]]:

        en_strs = [sample['translation']['en'] for sample in inputs]
        de_strs = [sample['translation']['de'] for sample in inputs]

        max_length = max([max([len(en) for en in en_strs]), max([len(de) for de in de_strs])])

        batched_en = self.tokenizer_en.batch_encode_plus(en_strs,
                                                         return_tensors='pt',
                                                         padding='max_length',
                                                         truncation=True,
                                                         max_length=min(max_length, self._config.max_sequence_len),
                                                         )
        batched_en.to(self.device)
        batched_de = self.tokenizer_en.batch_encode_plus(de_strs,
                                                         truncation=True,
                                                         return_tensors='pt',
                                                         padding='max_length',
                                                         max_length=min(max_length, self._config.max_sequence_len))
        batched_de.to(self.device)

        return {
            'en' : batched_en,
            'de' : batched_de,
        }

    def decode(self, tokens : Tensor, lang : str = 'de') -> list[str]:

        if lang == 'de':
            tokenizer = self.tokenizer_de
        elif lang == 'en':
            tokenizer = self.tokenizer_en
        else:
            raise ValueError(f'{lang} is not a supported language')

        # TODO: support also non-batch
        decoded = tokenizer.batch_decode(tokens)

        return decoded

def analyze_state_dict_shapes_and_names(model):
    # This part helped me figure out that I don't have positional encodings saved in the state dict
    print(model.state_dict().keys())

    # This part helped me see that src MHA was missing in the decoder since both it and trg MHA were referencing
    # the same MHA object in memory - stupid mistake, happens all the time, embrace the suck!
    for name, param in model.named_parameters():
        print(name, param.shape)
        if not param.requires_grad:
            raise Exception('Expected all of the params to be trainable - no param freezing used.')

# Count how many trainable weights the model has <- just for having a feeling for how big the model is
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import torch
    use_big_transformer = False

    # Dummy data
    src_vocab_size = 11
    trg_vocab_size = 11
    src_token_ids_batch = torch.randint(1, 10, size=(3, 2))
    trg_token_ids_batch = torch.randint(1, 10, size=(3, 2))

    cfg = Transformer.Config(
        model_dimension=512,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=trg_vocab_size,
        num_heads=8,
        num_decoder_blocks=6,
        num_encoder_blocks=6,
        dropout_prob=0.1,
        ff_dim=2048
    )

    transformer = Transformer(cfg)

    # These 2 functions helped me figure out the 2 bugs I had:
    # 1) I did not register positional encodings and thus they wouldn't be saved and later model-loading would fail
    # 2) I had a bug with MHA (attention) in decoder, where both src and trg were referencing the same MHA object in mem
    # It's a good practice to see whether the names, shapes and number of params make sense.
    # e.g. I knew that the big transformer had ~175 M params and I verified that here.
    analyze_state_dict_shapes_and_names(transformer)
    print(f'Size of the {"big" if use_big_transformer else "baseline"} transformer = {count_parameters(transformer)}')

    out = transformer(src_token_ids_batch, trg_token_ids_batch) #  src_mask=None, trg_mask=None)