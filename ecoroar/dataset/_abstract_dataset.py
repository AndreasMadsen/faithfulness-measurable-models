import pathlib
from functools import cached_property, partial
from typing import List, Tuple, Dict, Union, Iterable
from abc import ABCMeta, abstractmethod

import tensorflow as tf
import tensorflow_datasets as tfds

from ..tokenizer._abstract_tokenizer import AbstractTokenizer
from ..metric import AUROC, F1Score, Matthew, Pearson

class AbstractDataset(metaclass=ABCMeta):
    _name: str
    _metrics: List[str]
    _early_stopping_metric: str

    _split_train: str
    _split_valid: str
    _split_test: str

    _persistent_dir: pathlib.Path
    _builder_cache: tfds.core.DatasetBuilder
    _seed: int

    def __init__(self, persistent_dir: pathlib.Path, seed: int = 0, use_snapshot=True, use_cache=True):
        """Abstract Base Class for defining a dataset with standard train/valid/test semantics.

        Args:
            persistent_dir (pathlib.Path): Persistent directory, used for storing the dataset.
            seed (int, optional): Random seed used for initial shuffling. Defaults to 0.
            use_snapshot (int, optional): If the preprocessed dataset should be cached to disk.
            use_cache (int, optional): If the preprocessed dataset should be cached to memory.
        """

        self._persistent_dir = persistent_dir
        self._builder_cache = self._builder(data_dir=persistent_dir / 'cache' / 'tfds')
        self._seed = seed
        self._use_cache = use_cache
        self._use_snapshot = use_snapshot

    @abstractmethod
    def _builder(self, data_dir: pathlib.Path) -> tfds.core.DatasetBuilder:
        ...

    @abstractmethod
    def _as_supervised(self, item: Dict[str, tf.Tensor]) -> Tuple[Iterable[tf.Tensor], tf.Tensor]:
        x = (item['text'], )
        return x, item['label']

    def metrics(self) -> List[tf.keras.metrics.Metric]:
        """Return a list of metric which this dataset uses
        """
        possible_metric = {
            'accuracy': partial(tf.keras.metrics.SparseCategoricalAccuracy, name='accuracy'),
            'auroc': partial(AUROC, from_logits=True),
            'macro_f1': partial(F1Score, num_classes=self.num_classes, average='macro'),
            'micro_f1': partial(F1Score, num_classes=self.num_classes, average='micro'),
            'matthew': partial(Matthew, num_classes=self.num_classes)
        }

        return [
            possible_metric[metric_name]() for metric_name in self._metrics
        ]

    @property
    def early_stopping_metric(self) -> str:
        return f'val_{self._early_stopping_metric}'

    @property
    def name(self) -> str:
        """Name of the dataset
        """
        return self._name

    @property
    def info(self) -> tfds.core.DatasetInfo:
        """Standard information object
        """
        return self._builder_cache.info

    @property
    def num_classes(self) -> int:
        """Number of classes in the dataset
        """
        return self.info.features['label'].num_classes

    def download(self):
        """Downloads dataset
        """
        self._builder_cache.download_and_prepare(
            download_dir=self._persistent_dir / 'cache' / 'tfds'
        )

    @cached_property
    def _datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        return self._builder_cache.as_dataset(split=[self._split_train, self._split_valid, self._split_test],
                                              shuffle_files=True,
                                              read_config=tfds.ReadConfig(shuffle_seed=self._seed))

    def _snapshot_path(self, split: Union['train', 'valid', 'test'], tokenizer: AbstractTokenizer):
        dirname = self._persistent_dir / 'cache' / 'dataset_snapshot'
        if tokenizer:
            filename = f'd-{self.name}_s-{self._seed}_m-{tokenizer.name}.{split}.snapshot'
        else:
            filename = f'd-{self.name}_s-{self._seed}.{split}.snapshot'
        return dirname / filename

    def _preprocess(self, dataset: tf.data.Dataset, split: Union['train', 'valid', 'test'], tokenizer: AbstractTokenizer):
        dataset = dataset.map(self._as_supervised, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        if tokenizer:
            dataset = dataset.map(lambda x, y: (tokenizer(x), y), num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        if self._use_snapshot:
            dataset = dataset.snapshot(str(self._snapshot_path(split, tokenizer)))
        if self._use_cache:
            dataset = dataset.cache()
        return dataset

    @property
    def train_num_examples(self) -> int:
        """Number of training obsevations
        """
        return self.info.splits[self._split_train].num_examples

    def train(self, tokenizer: AbstractTokenizer=None) -> tf.data.Dataset:
        """Get training dataset
        """
        (train, _, _) = self._datasets
        return self._preprocess(train, 'train', tokenizer)

    @property
    def valid_num_examples(self) -> int:
        """Number of validation obsevations
        """
        return self.info.splits[self._split_valid].num_examples

    def valid(self, tokenizer: AbstractTokenizer=None) -> tf.data.Dataset:
        """Validation dataset
        """
        (_, valid, _) = self._datasets
        return self._preprocess(valid, 'valid', tokenizer)

    @property
    def test_num_examples(self) -> int:
        """Number of test obsevations
        """
        return self.info.splits[self._split_test].num_examples

    def test(self, tokenizer: AbstractTokenizer=None) -> tf.data.Dataset:
        """Test dataset
        """
        (_, _, test) = self._datasets
        return self._preprocess(test, 'test', tokenizer)
