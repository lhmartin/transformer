from torchtext.data.metrics import bleu_score
from torch import Tensor, argmax, zeros
from transformer import Transformer
from transformers import PreTrainedTokenizer
from typing import Dict


def decode_and_calculate_bleu_score(predictions : Tensor,
                                    targets : Tensor,
                                    tokenizer : PreTrainedTokenizer,
                                    ) -> float:

    pred_tokens = argmax(predictions, dim=-1)
    pred_tokens = pred_tokens.reshape(targets.shape)
    decoded_pred = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
    decoded_targ = tokenizer.batch_decode(targets, skip_special_tokens=True)

    decoded_pred_split = [sent.split(' ') for sent in decoded_pred]
    decoded_targ_split = [[sent.split(' ')] for sent in decoded_targ]

    return bleu_score(decoded_pred_split, decoded_targ_split)

def calculate_accuracy(predictions : Tensor, labels : Tensor, padding_idx : int = 0):

    labels = labels.flatten()
    ids = argmax(predictions, dim=-1)
    acc = ((ids == labels)[labels != padding_idx].float().sum() )/ (labels != 0).bool().sum()
    return acc

def one_hot_labels(labels : Tensor, vocab_size : int):

        batch_size = labels.shape[0]

        OH_tokens = zeros((batch_size, vocab_size), device=labels.device)
        OH_tokens.scatter_(1, labels, 1.0)
        OH_tokens[:, 0] = 0

        return OH_tokens

def greedy_decode_bleu_score(batch : Dict[str, Tensor],
                             src_lang : str,
                             trgt_lang : str,
                             model : Transformer,
                             trgt_tokenizer : PreTrainedTokenizer,
                             *,
                             multi_gpu_mode : bool = False
                             ):
    if multi_gpu_mode:
        model = model.module

    src_token_ids, trg_token_ids = batch[src_lang]['input_ids'], batch[trgt_lang]['input_ids']

    src_mask = model.make_source_mask(src_token_ids)
    src_representations_batch = model.encode(src_token_ids, src_mask)

    predicted_sentences = model.greedy_decoding(src_representations_batch,
                                                        src_mask,
                                                        trgt_tokenizer,
                                                        max_target_tokens=model._config.max_sequence_len)

    predicted_sentences_corpus = [[sent] for sent in predicted_sentences]  # add them to the corpus of translations

    # Get the token and not id version of GT (ground-truth) sentences
    gt_sentences_corpus = trgt_tokenizer.batch_decode(trg_token_ids.cpu(),
                                                                skip_special_tokens=True)  # add them to the corpus of GT translations

    return bleu_score(predicted_sentences_corpus, gt_sentences_corpus)
