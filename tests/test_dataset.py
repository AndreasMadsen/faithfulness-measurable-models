
import pathlib
from dataclasses import dataclass

import pytest
import tensorflow as tf
import numpy as np

from ecoroar.dataset import datasets
from ecoroar.tokenizer import HuggingfaceTokenizer


@dataclass
class DatasetSize:
    name: str
    train: int
    valid: int
    test: int
    num_classes: int


expected_sizes = [
    DatasetSize('bAbI-1', 8000, 2000, 1000, 6),
    DatasetSize('bAbI-2', 8000, 2000, 1000, 6),
    DatasetSize('bAbI-3', 8000, 2000, 1000, 6),
    DatasetSize('BoolQ', 7542, 1885, 3270, 2),
    DatasetSize('CB', 200, 50, 56, 3),
    DatasetSize('CoLA', 6841, 1710, 1043, 2),
    DatasetSize('IMDB', 20000, 5000, 25000, 2),
    DatasetSize('MNLI', 314162, 78540, 9815, 3),
    DatasetSize('MRPC', 2934, 734, 408, 2),
    DatasetSize('MIMIC-a', 4262, 729, 1243, 2),
    DatasetSize('MIMIC-d', 8066, 1573, 1729, 2),
    DatasetSize('QNLI', 83794, 20949, 5463, 2),
    DatasetSize('QQP', 291077, 72769, 40430, 2),
    DatasetSize('RTE', 1992, 498, 277, 2),
    DatasetSize('SNLI', 549367, 9842, 9824, 3),
    DatasetSize('SST2', 53879, 13470, 872, 2),
    DatasetSize('WNLI', 508, 127, 71, 2),
]


@pytest.mark.parametrize("info", expected_sizes, ids=lambda info: info.name)
def test_dataset_name(info):
    dataset = datasets[info.name](persistent_dir=pathlib.Path('.'), use_snapshot=False, use_cache=False)

    assert dataset.name == info.name


@pytest.mark.parametrize("info", expected_sizes, ids=lambda info: info.name)
def test_dataset_num_classes(info):
    dataset = datasets[info.name](persistent_dir=pathlib.Path('.'), use_snapshot=False, use_cache=False)

    assert dataset.num_classes == info.num_classes


@pytest.mark.parametrize("info", expected_sizes, ids=lambda info: info.name)
def test_dataset_num_examples(info):
    dataset = datasets[info.name](persistent_dir=pathlib.Path('.'), use_snapshot=False, use_cache=False)

    assert dataset.train_num_examples == info.train
    assert dataset.num_examples('train') == info.train
    assert dataset.valid_num_examples == info.valid
    assert dataset.num_examples('valid') == info.valid
    assert dataset.test_num_examples == info.test
    assert dataset.num_examples('test') == info.test


@pytest.mark.parametrize("info", expected_sizes, ids=lambda info: info.name)
def test_dataset_cadinality(info):
    dataset = datasets[info.name](persistent_dir=pathlib.Path('.'), use_snapshot=False, use_cache=False)

    assert dataset.train().cardinality() == info.train
    assert dataset.load('train').cardinality() == info.train
    assert dataset.valid().cardinality() == info.valid
    assert dataset.load('valid').cardinality() == info.valid
    assert dataset.test().cardinality() == info.test
    assert dataset.load('test').cardinality() == info.test


@pytest.mark.parametrize("info", expected_sizes, ids=lambda info: info.name)
@pytest.mark.slow
def test_tokenizer_integration(info):
    tokenizer = HuggingfaceTokenizer('roberta-base', persistent_dir=pathlib.Path('.'))
    dataset = datasets[info.name](persistent_dir=pathlib.Path('.'), use_snapshot=False, use_cache=False)

    for x_raw, y in dataset.valid().take(1):
        x_encoded = tokenizer(x_raw)
        tf.debugging.assert_shapes([
            (x_encoded['input_ids'], ('T', )),
            (x_encoded['attention_mask'], ('T', ))
        ])


@pytest.mark.parametrize("info", expected_sizes, ids=lambda info: info.name)
@pytest.mark.slow
def test_class_count(info):
    dataset = datasets[info.name](persistent_dir=pathlib.Path('.'), use_snapshot=False, use_cache=False, seed=0)

    def class_count(d):
        return d \
            .map(lambda x, y: y, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False) \
            .batch(4096, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False) \
            .reduce(
                tf.zeros((dataset.num_classes, ), dtype=tf.dtypes.int32),
                lambda r, y: r + tf.math.bincount(tf.cast(y, dtype=tf.dtypes.int32),
                                                  minlength=dataset.num_classes, maxlength=dataset.num_classes)
            ) \
            .numpy() \
            .tolist()

    train_class_count = class_count(dataset.train())
    assert train_class_count == dataset.train_class_count
    assert sum(train_class_count) == dataset.train_num_examples

    valid_class_count = class_count(dataset.valid())
    assert valid_class_count == dataset.valid_class_count
    assert sum(valid_class_count) == dataset.valid_num_examples

    test_class_count = class_count(dataset.test())
    assert test_class_count == dataset.test_class_count
    assert sum(test_class_count) == dataset.test_num_examples


@pytest.mark.parametrize("info", expected_sizes, ids=lambda info: info.name)
def test_majority_classifier_test_performance(info):
    Dataset = datasets[info.name]
    metrics = Dataset(persistent_dir=pathlib.Path('.'), use_snapshot=False, use_cache=False).metrics()
    performance = Dataset.majority_classifier_performance()

    class_count_test = np.asarray(Dataset._class_count_test)
    best_class_idx = np.argmax(Dataset._class_count_train)

    y_true = np.vstack([
        np.full((count, 1), label, dtype=np.int64)
        for label, count in enumerate(class_count_test)
    ])
    y_pred = np.zeros((np.sum(class_count_test), class_count_test.size), dtype=np.float32)
    y_pred[:, best_class_idx] = 1

    for metric in metrics:
        metric.update_state(y_true, y_pred)
        np.testing.assert_allclose(metric.result(), performance[metric.name], rtol=1e-6)
