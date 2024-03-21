from torch.utils.data import Dataset
from typing import Literal
from datasets import load_dataset

DATASET_SPLITS = Literal['train', 'test', 'valid']
LANGUAGE_PAIRS = Literal['iwslt2017-ar-en', 'iwslt2017-de-en', 'iwslt2017-en-ar', 'iwslt2017-en-de', 'iwslt2017-en-fr',
                         'iwslt2017-en-it', 'iwslt2017-en-ja', 'iwslt2017-en-ko', 'iwslt2017-en-nl', 'iwslt2017-en-ro',
                         'iwslt2017-en-zh', 'iwslt2017-fr-en', 'iwslt2017-it-en', 'iwslt2017-it-nl', 'iwslt2017-it-ro',
                         'iwslt2017-ja-en', 'iwslt2017-ko-en', 'iwslt2017-nl-en', 'iwslt2017-nl-it', 'iwslt2017-nl-ro',
                         'iwslt2017-ro-en', 'iwslt2017-ro-it', 'iwslt2017-ro-nl', 'iwslt2017-zh-en']

class WMT14_Dataset(Dataset):

    def __init__(self,
                 split           : DATASET_SPLITS,
                 language_pair   : LANGUAGE_PAIRS = 'iwslt2017-de-en',
                 streaming       : bool           = False,
                 ) -> None:
        super().__init__()

        self.data = load_dataset("iwslt2017",
                                language_pair,
                                split=split,
                                streaming=streaming,
                                trust_remote_code=True)
    def __getitem__(self, idx : int):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    dataset = WMT14_Dataset(split='train')
    dataset[0]
    len(dataset)
    print()