from torchtext.data.metrics import bleu_score
from typing import List
from torch import Tensor, argmax
from transformers import PreTrainedTokenizer

def calculate_bleu_score(predictions : List[str], targets : List[str]) -> float:

    return bleu_score(predictions, targets)


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

    return calculate_bleu_score(decoded_pred_split, decoded_targ_split)

def calculate_accuracy(predictions : Tensor, labels : Tensor, padding_idx : int = 0):

    labels = labels.flatten()
    ids = argmax(predictions, dim=-1)
    acc = ((ids == labels)[labels != padding_idx].float().sum() )/ (labels != 0).bool().sum()
    return acc