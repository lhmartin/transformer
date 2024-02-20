from torch.utils.data import Dataset
from datasets import load_dataset, DatasetDict
from typing import Literal

LANGUAGE_PAIRS = Literal['de-en', 'cs-en', 'fr-en', 'hi-en', 'hu-en']
DATASET_SPLITS = Literal['train', 'test', 'validation']

class WMT14_Dataset(Dataset):

    def __init__(self, 
                 split         : DATASET_SPLITS,
                 language_pair : LANGUAGE_PAIRS = 'de-en',
                 streaming     : bool           = False,
                 ) -> None:
        super().__init__()

        self.data : DatasetDict = load_dataset('wmt14',
                                                  language_pair,
                                                  split=split,
                                                  streaming=streaming,
                                                  trust_remote_code=True)

    def __getitem__(self, idx : int):

        return self.data[0]

    def __len__(self):

        return len(self.data)


if __name__ == '__main__':

    ds = WMT14_Dataset(split='train')

    print('------------------------------')
    print(ds)
    print(len(ds))
    print(ds[0])
    print('------------------------------')