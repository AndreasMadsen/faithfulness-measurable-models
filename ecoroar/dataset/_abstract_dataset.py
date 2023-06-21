import pathlib
import shutil
from functools import cached_property, partial
from typing import List, Tuple, Dict, Union, Iterable, Optional
from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from ..types import Tokenizer
from ..metric import AUROC, F1Score, Matthew, Pearson

class AbstractDataset(metaclass=ABCMeta):
    _name: str
    _metrics: List[str]
    _early_stopping_metric: str
    _jain_etal_metric: Optional[str]
    _target_name: str = 'label'
    _convergence_threshold = -1  # A very lower-bound to check if the model convereged

    _split_train: str
    _split_valid: str
    _split_test: str

    _persistent_dir: pathlib.Path
    _builder_cache: tfds.core.DatasetBuilder
    _seed: int

    _class_count_train: List[int]
    _class_count_valid: List[int]
    _class_count_test: List[int]

    _input_masked: str
    _input_aux: Optional[str] = None

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
        self._use_snapshot = use_snapshot
        self._use_cache = use_cache

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

    @classmethod
    def majority_classifier_performance(cls, split: str = 'test'):
        class_count_train = np.asarray(cls._class_count_train)
        class_count_measure = np.asarray(getattr(cls, f'_class_count_{split}'))
        best_class_idx = np.argmax(class_count_train)
        num_classes = class_count_train.size

        # Some notation from https://en.wikipedia.org/wiki/Phi_coefficient
        c = class_count_measure[best_class_idx] # total number of smaples correctly predicted
        s = np.sum(class_count_measure) # total number of samples

        possible_metric = {
            'loss': np.nan,  # Not possible to compute cross entropy of zero probability
            'accuracy': c / s,
            'auroc': None,  # Can not be meaningfully computed without continues values
            'macro_f1': (1/num_classes) * c / (0.5*s + 0.5*c),  # One class, will have a non-zero score
            'micro_f1': c / s,  # Micro is always accuracy, for a majority classifier
            'matthew': 0  # Mathhew is always zero, for majority classifier
        }

        return {
            metric_name: possible_metric[metric_name] for metric_name in cls._metrics + ['loss']
        }

    @classmethod
    def summary(cls):
        return {
            'name': cls._name,
            'metric': cls._early_stopping_metric,
            'baseline': cls.majority_classifier_performance('test')[cls._early_stopping_metric],
            'train': sum(cls._class_count_train),
            'valid': sum(cls._class_count_valid),
            'test': sum(cls._class_count_test),
            'masked': cls._input_masked,
            'auxilary': cls._input_aux
        }

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
        return self.info.features[self._target_name].num_classes

    @property
    def class_names(self) -> List[str]:
        """Class names
        """
        return self.info.features[self._target_name].names

    def download(self):
        """Downloads dataset
        """
        self._builder_cache.download_and_prepare(
            download_dir=self._persistent_dir / 'cache' / 'tfds',
            download_config=tfds.download.DownloadConfig(
                manual_dir=self._persistent_dir
            )
        )

    @cached_property
    def _dataset_index(self) -> Dict[str, int]:
        return {
            'train': 0,
            'valid': 1,
            'test': 2
        }

    @cached_property
    def _datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        return self._builder_cache.as_dataset(split=[self._split_train, self._split_valid, self._split_test],
                                              shuffle_files=True,
                                              read_config=tfds.ReadConfig(shuffle_seed=self._seed))

    def _preprocess_path(self, split: Union['train', 'valid', 'test'], tokenizer: Tokenizer=None):
        dirname = self._persistent_dir / 'cache' / 'dataset_preprocess'
        if tokenizer:
            filename = f'd-{self.name.lower()}_s-{self._seed}_t-{tokenizer.alias_name.lower()}.{split}.tfds'
        else:
            filename = f'd-{self.name.lower()}_s-{self._seed}.{split}.tfds'
        return dirname / filename

    def _process_dataset(self, dataset: tf.data.Dataset, tokenizer: Tokenizer=None):
        # process dataset
        dataset = dataset.map(self._as_supervised, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        if tokenizer:
            dataset = dataset.map(lambda x, y: (tokenizer(x), y), num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

        return dataset

    def preprocess(self, tokenizer: Tokenizer=None):
        """Creates preprocessed datasets, for each train, valid, and test split.

        These datasets uses tf.data.Dataset.save as the storage format, and generally loads much faster.

        Args:
            tokenizer (Tokenizer, optional): If provided, the datasets will be tokenized. Defaults to None.
        """
        for split, dataset in zip(['train', 'valid', 'test'], self._datasets):
            # save dataset
            path = self._preprocess_path(split, tokenizer)
            if path.exists():
                shutil.rmtree(path)
            dataset = self._process_dataset(dataset, tokenizer)
            dataset.save(str(path))

    def is_preprocess_valid(self, tokenizer: Tokenizer=None):
        for name, split in [('train', self._split_train), ('valid', self._split_valid), ('test', self._split_test)]:
            path = self._preprocess_path(name, tokenizer)
            if not path.exists():
                print(f'File missing mismatch: {path}')
                return False
            dataset = tf.data.Dataset.load(str(path))
            if dataset.cardinality() != self.info.splits[split].num_examples:
                print(f'Cadinality mismatch: {dataset.cardinality()} {self.info.splits[split].num_examples}')
                return False
        return True

    def load(self, split: Union['train', 'valid', 'test'], tokenizer: Tokenizer=None):
        if self._use_snapshot:
            path = self._preprocess_path(split, tokenizer)
            if not path.exists():
                raise IOError('preprocessed dataset does not exist, call dataset.preprocess(tokenizer)')

            dataset = tf.data.Dataset.load(str(path))
        else:
            dataset = self._process_dataset(self._datasets[self._dataset_index[split]], tokenizer)

        if self._use_cache:
            dataset = dataset.cache()

        return dataset

    def num_examples(self, split: Union['train', 'valid', 'test']):
        match split:
            case 'train':
                return self.train_num_examples
            case 'valid':
                return self.valid_num_examples
            case 'test':
                return self.test_num_examples
            case _:
                raise ValueError(f'split {split} is not supported')

    @property
    def train_num_examples(self) -> int:
        """Number of training obsevations
        """
        return self.info.splits[self._split_train].num_examples

    @property
    def train_class_count(self) -> List[int]:
        """Number of training obsevations
        """
        return self._class_count_train.copy()

    def train(self, tokenizer: Tokenizer=None) -> tf.data.Dataset:
        """Get training dataset
        """
        return self.load('train', tokenizer)

    @property
    def valid_num_examples(self) -> int:
        """Number of validation obsevations
        """
        return self.info.splits[self._split_valid].num_examples

    @property
    def valid_class_count(self) -> List[int]:
        """Number of validation obsevations
        """
        return self._class_count_valid.copy()

    def valid(self, tokenizer: Tokenizer=None) -> tf.data.Dataset:
        """Validation dataset
        """
        return self.load('valid', tokenizer)

    @property
    def test_num_examples(self) -> int:
        """Number of test obsevations
        """
        return self.info.splits[self._split_test].num_examples

    @property
    def test_class_count(self) -> List[int]:
        """Number of test obsevations
        """
        return self._class_count_test.copy()

    def test(self, tokenizer: Tokenizer=None) -> tf.data.Dataset:
        """Test dataset
        """
        return self.load('test', tokenizer)
