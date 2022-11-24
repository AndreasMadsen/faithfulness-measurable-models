import pathlib
import shutil
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
        ...

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

    def _preprocess_path(self, split: Union['train', 'valid', 'test'], tokenizer: AbstractTokenizer=None):
        dirname = self._persistent_dir / 'cache' / 'dataset_preprocess'
        if tokenizer:
            filename = f'd-{self.name}_s-{self._seed}_t-{tokenizer.alias_name}.{split}.tfds'
        else:
            filename = f'd-{self.name}_s-{self._seed}.{split}.tfds'
        return dirname / filename

    def preprocess(self, tokenizer: AbstractTokenizer=None):
        """Creates preprocessed datasets, for each train, valid, and test split.

        These datasets uses tf.data.Dataset.save as the storage format, and generally loads much faster.

        Args:
            tokenizer (AbstractTokenizer, optional): If provided, the datasets will be tokenized. Defaults to None.
        """
        for split, dataset in zip(['train', 'valid', 'test'], self._datasets):
            # process dataset
            dataset = dataset.map(self._as_supervised, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
            if tokenizer:
                dataset = dataset.map(lambda x, y: (tokenizer(x), y), num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

            # save dataset
            path = self._preprocess_path(split, tokenizer)
            if path.exists():
                shutil.rmtree(path)
            dataset.save(str(path))

    def _load_preprocess(self, split: Union['train', 'valid', 'test'], tokenizer: AbstractTokenizer=None):
        path = self._preprocess_path(split, tokenizer)
        if not path.exists():
            raise IOError('preprocessed dataset does not exist, call dataset.preprocess(tokenizer)')
        return tf.data.Dataset.load(str(path))

    @property
    def train_num_examples(self) -> int:
        """Number of training obsevations
        """
        return self.info.splits[self._split_train].num_examples

    def train(self, tokenizer: AbstractTokenizer=None) -> tf.data.Dataset:
        """Get training dataset
        """
        return self._load_preprocess('train', tokenizer).cache()

    @property
    def valid_num_examples(self) -> int:
        """Number of validation obsevations
        """
        return self.info.splits[self._split_valid].num_examples

    def valid(self, tokenizer: AbstractTokenizer=None) -> tf.data.Dataset:
        """Validation dataset
        """
        return self._load_preprocess('valid', tokenizer).cache()

    @property
    def test_num_examples(self) -> int:
        """Number of test obsevations
        """
        return self.info.splits[self._split_test].num_examples

    def test(self, tokenizer: AbstractTokenizer=None) -> tf.data.Dataset:
        """Test dataset
        """
        return self._load_preprocess('test', tokenizer).cache()
