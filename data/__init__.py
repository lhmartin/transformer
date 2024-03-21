from typing import Literal

DATASETS = Literal['WMT14', 'ILSWT17']
DATASET_SPLITS = ['train', 'test', 'validation']

from .ilswt17_dataset import ILSWT17_Dataset
from .wmt14_dataset import WMT14_Dataset