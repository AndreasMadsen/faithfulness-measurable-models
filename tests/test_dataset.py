
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
    DatasetSize('BoolQ', 9427, 3270, 3245),
    DatasetSize('CoLA', 8551, 1043, 1063),
    DatasetSize('IMDB', 20000, 5000, 25000),
    DatasetSize('MNLI', 392702, 9815, 9796),
    DatasetSize('QQP', 363846, 40430, 390965),
    DatasetSize('SST2', 67349, 872, 1821),
]

@pytest.mark.parametrize("info", expected_sizes, ids=lambda info: info.name)
def test_dataset_size(info):
    dataset = datasets[info.name](persistent_dir=pathlib.Path('.'))

    assert dataset.name == info.name
    assert dataset.train_num_examples == info.train
    assert dataset.valid_num_examples == info.valid
    assert dataset.test_num_examples == info.test
