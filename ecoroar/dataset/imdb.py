
from functools import cached_property
from typing import Tuple

import tensorflow as tf
import tensorflow_datasets as tfds


class IMDBDataset:
    def __init__(self, persistent_dir: str, seed: int = 0):
        """Create IMDB dataset

        Args:
            persistent_dir (str): Persistent directory, used for storing the dataset
            seed (int, optional): Random seed used for initial shuffling. Defaults to 0.
        """
        self._persistent_dir = persistent_dir
        self._builder = tfds.builder("imdb_reviews", try_gcs=False, data_dir=f'{persistent_dir}/download/tfds/')
        self._seed = seed

    @property
    def name(self) -> str:
        """Name of the dataset
        """
        return "imdb"

    @property
    def info(self) -> tfds.core.DatasetInfo:
        """Standard information object
        """
        return self._builder.info

    @property
    def num_classes(self) -> int:
        """Number of classes in the dataset
        """
        return self.info.features['label'].num_classes

    def download(self):
        """Downloads dataset
        """
        self._builder.download_and_prepare(download_dir=f'{self._persistent_dir}/download/tfds/downloads/')

    @cached_property
    def _datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        return self._builder.as_dataset(split=['train[:80%]', 'train[80%:]', 'test'],
                                        shuffle_files=True,
                                        read_config=tfds.ReadConfig(shuffle_seed=self._seed))

    @property
    def train_num_examples(self) -> int:
        """Number of training obsevations
        """
        return self.info.splits['train[:80%]'].num_examples

    @property
    def train(self) -> tf.data.Dataset:
        """Get training dataset
        """
        (train, _, _) = self._datasets
        return train

    @property
    def valid_num_examples(self) -> int:
        """Number of validation obsevations
        """
        return self.info.splits['train[80%:]'].num_examples

    @property
    def valid(self) -> tf.data.Dataset:
        """Validation dataset
        """
        (_, valid, _) = self._datasets
        return valid

    @property
    def test_num_examples(self) -> int:
        """Number of test obsevations
        """
        return self.info.splits['test'].num_examples

    @property
    def test(self) -> tf.data.Dataset:
        """Test dataset
        """
        (_, _, test) = self._datasets
        return test
