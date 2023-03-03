
import pathlib
from dataclasses import dataclass

import pytest
import tensorflow as tf

from ecoroar.dataset import datasets
from ecoroar.tokenizer import HuggingfaceTokenizer

@dataclass
class DatasetSize:
    name: str
    train: int
    valid: int
    test: int

expected_sizes = [
    DatasetSize('bAbI-1', 8000, 2000, 1000),
    DatasetSize('bAbI-2', 8000, 2000, 1000),
    DatasetSize('bAbI-3', 8000, 2000, 1000),
    DatasetSize('BoolQ', 7542, 1885, 3270),
    DatasetSize('CB', 200, 50, 56),
    DatasetSize('CoLA', 6841, 1710, 1043),
    DatasetSize('IMDB', 20000, 5000, 25000),
    DatasetSize('MNLI', 314162, 78540, 9815),
    DatasetSize('MRPC', 2934, 734, 408),
    DatasetSize('QNLI', 83794, 20949, 5463),
    DatasetSize('QQP', 291077, 72769, 40430),
    DatasetSize('RTE', 1992, 498, 277),
    DatasetSize('SST2', 53879, 13470, 872),
    DatasetSize('WNLI', 508, 127, 71),
]

@pytest.mark.parametrize("info", expected_sizes, ids=lambda info: info.name)
def test_dataset_name(info):
    dataset = datasets[info.name](persistent_dir=pathlib.Path('.'), use_snapshot=False, use_cache=False)

    assert dataset.name == info.name

@pytest.mark.parametrize("info", expected_sizes, ids=lambda info: info.name)
def test_dataset_size(info):
    dataset = datasets[info.name](persistent_dir=pathlib.Path('.'), use_snapshot=False, use_cache=False)

    assert dataset.train_num_examples == info.train
    assert dataset.valid_num_examples == info.valid
    assert dataset.test_num_examples == info.test

@pytest.mark.parametrize("info", expected_sizes, ids=lambda info: info.name)
def test_tokenizer_integration(info):
    tokenizer = HuggingfaceTokenizer('roberta-base', persistent_dir=pathlib.Path('.'))
    dataset = datasets[info.name](persistent_dir=pathlib.Path('.'), use_snapshot=False, use_cache=False)

    for x_raw, y in dataset.valid().take(1):
        x_encoded = tokenizer(x_raw)
        tf.debugging.assert_shapes([
            (x_encoded['input_ids'], ('T', )),
            (x_encoded['attention_mask'], ('T', ))
        ])
