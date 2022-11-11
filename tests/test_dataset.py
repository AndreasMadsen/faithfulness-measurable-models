
import pathlib
from dataclasses import dataclass

import pytest

from ecoroar.dataset import datasets

@dataclass
class DatasetSize:
    name: str
    train: int
    valid: int
    test: int

expected_sizes = [
    DatasetSize('BoolQ', 7542, 1885, 3270),
    DatasetSize('CoLA', 6841, 1710, 1043),
    DatasetSize('IMDB', 20000, 5000, 25000),
    DatasetSize('MNLI', 314162, 78540, 9815),
    DatasetSize('QQP', 291077, 72769, 40430),
    DatasetSize('SST2', 53879, 13470, 872),
]

@pytest.mark.parametrize("info", expected_sizes, ids=lambda info: info.name)
def test_dataset_size(info):
    dataset = datasets[info.name](persistent_dir=pathlib.Path('.'))

    assert dataset.name == info.name
    assert dataset.train_num_examples == info.train
    assert dataset.valid_num_examples == info.valid
    assert dataset.test_num_examples == info.test
