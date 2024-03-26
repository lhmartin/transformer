from typing import Dict, Literal, Tuple, List
import torch.nn as nn
from torch import Tensor, argmax, triu, ones, tensor, cat, unsqueeze

from modules.positional_encoding import SinCosPositionalEmbedding
from modules.encode_decode_blocks import EncoderBlock, DecoderBlock
from pydantic import BaseModel
from transformers import AutoTokenizer
from math import sqrt


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
            num_embeddings=self._config.src_vocab_size,
            embedding_dim=self._config.model_dimension,
        )
        self.decoder_embedder = nn.Embedding(
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
        self.log_softmax = nn.LogSoftmax(-1)
        self.dropout = nn.Dropout(self._config.dropout_prob)

        self.tokenizer_en = AutoTokenizer.from_pretrained('bert-base-cased')
        self.tokenizer_de = AutoTokenizer.from_pretrained('bert-base-german-cased')

    def encode(self, input_sequence: Tensor, mask : Tensor | None = None) -> Tensor:
        """Embed the the input sequence using the embedding trunk.

        Args:
            input_sequence (Tensor): Sequence of token ids of shape: [batch_size , seq_len]
            mask (Tensor | None, optional):
                Mask of where the padding tokens are shape [batch_size , 1 , 1 , seq_len].
                It has this shape, because it will be used on the attention matrix.
                which has shape: [batch_size , num_heads , seq_len , seq_len], so they must match in the
                number of dimensions to be applied to each other.
                Defaults to None.

        Returns:
            Tensor: A matrix of the embedded sequence. Shape: [batch_size , seq_len , model_dimension]
        """

        # Embed the single ids into learned embeddings: [batch_size , seq_len] -> [batch_size , seq_len , model_dimension]
        tokens = self.source_embedder(input_sequence) * sqrt(self._config.model_dimension)

        # add positional embeddings, these encode the relative positions of the tokens
        # allowing the model to understand the order in which the tokens occur.
        tokens = self.pos_encoding_enc(tokens)

        tokens = self.dropout(tokens)

        # loop through each encoding block and then return the final set of embedded tokens
        for encode_block in self.encoder_trunk:
            tokens = encode_block(tokens, mask)

        return tokens

    def decode(self,
               input_embeddings: Tensor,
               target: Tensor,
               src_mask : Tensor | None = None,
               trg_mask : Tensor | None = None) -> Tensor:
        """Decode the embbeded input to a set of logits representing the model's prediction for the
        most likely token at each position in the batch.

        Args:
            input_embeddings (Tensor):
                The matrix of embeddings representing the models embedded understanding
                of the input seqeunce. Shape : [batch_size , seq_len , model_dimension]
            target (Tensor):
                The input_ids of the target tokens. Shape : [batch_size , seq_len]
            src_mask (Tensor | None, optional):
                This mask shows which portions of the input embeddings
                correspond to padding tokens. Shape: [batch_size , 1 , 1 , seq_len].
                Defaults to None.
            trg_mask (Tensor | None, optional):
                This mask stops the model from looking forward at the target tokens, and stops
                the model from attending to padding tokens
                Shape: [batch_size, 1, seq_len,  seq_len].
                Defaults to None.

        Returns:
            Tensor: A matrix of log probabilities for each target token.
            Shape: [batch_size * seq_len, target_vocab_size]
        """

        # First encode the target tokens into embedings and then add positional encodings
        decode_tkns = self.decoder_embedder(target)
        decode_tkns = self.pos_encoding_dec(decode_tkns)
        decode_tkns = self.dropout(decode_tkns)

        # Pass through each of the decoding blocks
        for decode_block in self.decoder_trunk:
            decode_embeddings = decode_block(decode_tkns, input_embeddings, src_mask, trg_mask)

        # Take the decode embeddings and expand out to the target vocab size
        # [batch_size, seq_len, model_dimension] -> [batch_size, seq_len, target_vocab_size]
        logits = self.final_linear(decode_embeddings)

        # Take the log softmax over the last dimension, turning the logits in to log probabilities
        logits = self.log_softmax(logits)

        # reshape from [batch_size, seq_len, target_vocab_size] to [batch_size * seq_len, target_vocab_size]
        # because this is the expected shape for the loss function.
        return logits.reshape(-1, logits.shape[-1])

    def make_target_mask(self, target_tkns : Tensor):

        batch_size, seq_len    = target_tkns.size(0), target_tkns.size(1)

        trgt_padding_mask = (target_tkns != self.tokenizer_de.pad_token_id).view(batch_size, 1, 1, -1)

        look_ahead_mask = (1 - triu(ones(1, 1, seq_len,seq_len, device=self.device), diagonal=1)).bool()
        return (look_ahead_mask & trgt_padding_mask)

    def make_source_mask(self, input_tkns : Tensor):
        input_padding_mask = (input_tkns != self.tokenizer_en.pad_token_id)

        return input_padding_mask.unsqueeze(1).unsqueeze(1)

    def make_masks(self, input_tkns : Tensor, target_tkns : Tensor) -> Tuple[Tensor, Tensor]:

        src_mask = self.make_source_mask(input_tkns)
        trgt_mask = self.make_target_mask(target_tkns)

        return src_mask, trgt_mask

    def forward(self, input_tkns : Tensor, target_tkns : Tensor) -> Tensor:
        """Predict the most likely tokens to follow at each position of target_tkns.

        Args:
            input_tkns (Tensor):
                A set of input_ids representing the sequence that will be translated from.
                shape : [batch_size, seq_len]
            target_tkns (Tensor):
                A set of input_ids representing of the target sequence.
                shape : [batch_size, seq_len]

        Returns:
            Tensor: A matrix of log probabilities for each target token.
            Shape: [batch_size * seq_len, target_vocab_size]
        """

        src_mask, trg_mask = self.make_masks(input_tkns, target_tkns)

        input_embeddings = self.encode(input_tkns, mask=src_mask)

        return self.decode(input_embeddings=input_embeddings,
                           target=target_tkns,
                           src_mask=src_mask,
                           trg_mask=trg_mask)

    def init_params(self, weight_init : Literal['default', 'xavier']):

        if weight_init == 'xavier':
            for _, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def config(self):
        return self._config

    def collate_fn(self, data_dict : Dict[str, str]) -> Dict[str, Tensor]:
        """Transforms the strings from the dataset into form that is usable by the model.

        Args:
            inputs (Dict[str, str]):
                A dictionary containing strings of the input and target sentences.

        Returns:
            Dict[str, Tensor]: _description_
        """

        en_strs = [sample['translation']['en'] for sample in data_dict]
        de_strs = [sample['translation']['de'] for sample in data_dict]

        max_length = max([max([len(en) for en in en_strs]), max([len(de) for de in de_strs])])

        batched_en = self.tokenizer_en.batch_encode_plus(en_strs,
                                                         return_tensors='pt',
                                                         padding='max_length',
                                                         truncation=True,
                                                         max_length=min(max_length, self._config.max_sequence_len),
                                                         )
        batched_en.to(self.device)
        batched_de = self.tokenizer_de.batch_encode_plus(de_strs,
                                                         truncation=True,
                                                         return_tensors='pt',
                                                         padding='max_length',
                                                         max_length=min(max_length, self._config.max_sequence_len))
        batched_de.to(self.device)

        return {
            'en' : batched_en,
            'de' : batched_de,
        }

    def inference(self, texts_to_translate : List[str], direction : str = 'de_to_en') -> str:

        if direction == 'de_to_en':
            input_tokens = self.tokenizer_de.batch_encode_plus(texts_to_translate, return_tensors='pt')['input_ids']
            trg_tokenizer = self.tokenizer_en
        elif direction == 'en_to_de':
            input_tokens = self.tokenizer_en.batch_encode_plus(texts_to_translate, return_tensors='pt')['input_ids']
            trg_tokenizer = self.tokenizer_de

        src_mask = self.make_source_mask(input_tokens)
        src_embeddings = self.encode(input_tokens, src_mask)

        return self.greedy_decoding(src_embeddings, src_mask=src_mask, trg_field_processor=trg_tokenizer)

    def greedy_decoding(self, src_representations_batch, src_mask, trg_field_processor, max_target_tokens=100):
        """
        Supports batch (decode multiple source sentences) greedy decoding.

        Adapted from https://github.com/gordicaleksa/pytorch-original-transformer/tree/d5b29a41c5c3f68e1bcd0c528a58281632fa9d6d
        """

        device = next(self.parameters()).device
        bos_token = trg_field_processor.cls_token

        # Initial prompt is the beginning/start of the sentence token. Make it compatible shape with source batch => (B,1)
        target_sentences_tokens = [[bos_token] for _ in range(src_representations_batch.shape[0])]
        trg_token_ids_batch = tensor([[trg_field_processor.vocab[tokens[0]]] for tokens in target_sentences_tokens], device=device)

        # Set to true for a particular target sentence once it reaches the EOS (end-of-sentence) token
        is_decoded = [False] * src_representations_batch.shape[0]

        while True:
            trg_mask = self.make_target_mask(trg_token_ids_batch)
            # Shape = (B*T, V) where T is the current token-sequence length and V target vocab size
            predicted_log_distributions = self.decode(src_representations_batch, trg_token_ids_batch, src_mask, trg_mask)

            # Extract only the indices of last token for every target sentence (we take every T-th token)
            num_of_trg_tokens = len(target_sentences_tokens[0])
            predicted_log_distributions = predicted_log_distributions[num_of_trg_tokens-1::num_of_trg_tokens]

            # This is the "greedy" part of the greedy decoding:
            # We find indices of the highest probability target tokens and discard every other possibility
            most_probable_last_token_indices = argmax(predicted_log_distributions, dim=-1).cpu().numpy()

            # Find target tokens associated with these indices
            predicted_words = trg_field_processor.convert_ids_to_tokens(most_probable_last_token_indices)

            for idx, predicted_word in enumerate(predicted_words):
                target_sentences_tokens[idx].append(predicted_word)

                if predicted_word == trg_field_processor.sep_token:  # once we find EOS token for a particular sentence we flag it
                    is_decoded[idx] = True

            if all(is_decoded) or num_of_trg_tokens == max_target_tokens:
                break

            # Prepare the input for the next iteration (merge old token ids with the new column of most probable token ids)
            trg_token_ids_batch = cat((trg_token_ids_batch, unsqueeze(tensor(most_probable_last_token_indices, device=device), 1)), 1)

        # Post process the sentences - remove everything after the EOS token
        target_sentences_tokens_post = trg_field_processor.batch_decode(trg_token_ids_batch, skip_special_tokens=True)

        return target_sentences_tokens_post