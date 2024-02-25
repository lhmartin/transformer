from torchtext.data.metrics import bleu_score
from typing import List

def calculate_bleu_score(predictions : List[str], targets : List[str]) -> float:

    return bleu_score(predictions, targets)