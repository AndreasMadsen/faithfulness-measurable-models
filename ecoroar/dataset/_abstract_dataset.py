import tensorflow as tf
import tensorflow_datasets as tfds

from functools import cached_property
from typing import Tuple, Dict
import pathlib
from abc import ABCMeta, abstractmethod


class AbstractDataset(metaclass=ABCMeta):
    _name: str
    _metric: str

    _split_train: str
    _split_valid: str
    _split_test: str

    _persistent_dir: pathlib.Path
    _builder_cache: tfds.core.DatasetBuilder
    _seed: int

    def __init__(self, persistent_dir: pathlib.Path, seed: int = 0):
        """Abstract Base Class for defining a dataset with standard train/valid/test semantics.

        Args:
            persistent_dir (pathlib.Path): Persistent directory, used for storing the dataset.
            seed (int, optional): Random seed used for initial shuffling. Defaults to 0.
        """

        self._persistent_dir = persistent_dir
        self._builder_cache = self._builder(data_dir=persistent_dir / 'cache' / 'tfds')
        self._seed = seed

    @abstractmethod
    def _builder(self, data_dir: pathlib.Path) -> tfds.core.DatasetBuilder:
        ...

    @abstractmethod
    def _as_supervised(self, item: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        x = (item['text'], )
        return x, item['label']

    @property
    def metric(self) -> str:
        """The test metric to use
        """
        return self._metric

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

    @property
    def train_num_examples(self) -> int:
        """Number of training obsevations
        """
        return self.info.splits[self._split_train].num_examples

    @property
    def train(self) -> tf.data.Dataset:
        """Get training dataset
        """
        (train, _, _) = self._datasets
        return train.map(self._as_supervised, num_parallel_calls=tf.data.AUTOTUNE)

    @property
    def valid_num_examples(self) -> int:
        """Number of validation obsevations
        """
        return self.info.splits[self._split_valid].num_examples

    @property
    def valid(self) -> tf.data.Dataset:
        """Validation dataset
        """
        (_, valid, _) = self._datasets
        return valid.map(self._as_supervised, num_parallel_calls=tf.data.AUTOTUNE)

    @property
    def test_num_examples(self) -> int:
        """Number of test obsevations
        """
        return self.info.splits[self._split_test].num_examples

    @property
    def test(self) -> tf.data.Dataset:
        """Test dataset
        """
        (_, _, test) = self._datasets
        return test.map(self._as_supervised, num_parallel_calls=tf.data.AUTOTUNE)
