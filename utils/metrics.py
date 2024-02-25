from torchtext.data.metrics import bleu_score
from typing import List
from torch import Tensor
from transformers import PreTrainedTokenizer

def calculate_bleu_score(predictions : List[str], targets : List[str]) -> float:

    return bleu_score(predictions, targets)


def decode_and_calculate_bleu_score(predictions : Tensor,
                                    targets : Tensor,
                                    tokenizer : PreTrainedTokenizer,
                                    ) -> float:

    decoded_pred = tokenizer.batch_decode(predictions)
    decoded_targ = tokenizer.batch_decode(targets)

    return calculate_bleu_score(decoded_pred, decoded_targ)

def calculate_accuracy(predictions : Tensor, labels : Tensor):

    ids = argmax(predictions, dim=-1)
    acc = ((ids == labels)[labels != 0].float().sum() )/ (labels != 0).bool().sum()