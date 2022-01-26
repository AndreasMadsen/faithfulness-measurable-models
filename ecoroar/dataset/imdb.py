
from functools import cached_property

import tensorflow as tf
import tensorflow_datasets as tfds

class IMDBDataset:
    def __init__(self, persistent_dir, seed=0):
        self._persistent_dir = persistent_dir
        self._builder = tfds.builder("imdb_reviews", try_gcs=False, data_dir=f'{persistent_dir}/cache/tfds/')
        self._seed = seed

    @property
    def info(self):
        return self._builder.info

    @property
    def num_classes(self):
        return self.info.features['label'].num_classes

    def download(self):
        self._builder.download_and_prepare(download_dir=f'{self._persistent_dir}/cache/tfds/downloads/')

    @cached_property
    def datasets(self):
        return self._builder.as_dataset(split=['train[:80%]', 'train[80%:]', 'test'],
                                        shuffle_files=True,
                                        read_config=tfds.ReadConfig(shuffle_seed=self._seed))

    @property
    def train_num_examples(self):
        return self.info.splits['train[:80%]'].num_examples

    @property
    def train(self):
        (train, _, _) = self.datasets
        return train

    @property
    def valid_num_examples(self):
        return self.info.splits['train[80%:]'].num_examples

    @property
    def valid(self):
        (_, valid, _) = self.datasets
        return valid

    @property
    def test_num_examples(self):
        return self.info.splits['test'].num_examples

    @property
    def test(self):
        (_, _, test) = self.datasets
        return test
