import tensorflow_datasets as tfds

from ._abstract_dataset import AbstractDataset


class MultiNLIDataset(AbstractDataset):
    _name = 'MultiNLI'
    _metric = 'acc'

    _split_train = 'train[:80%]'
    _split_valid = 'train[80%:]'
    _split_test = 'validation_matched'

    def _builder(self, data_dir) -> tfds.core.DatasetBuilder:
        return tfds.builder("multi_nli", data_dir=data_dir)

    def _as_supervised(self, item):
        x = (item['premise'], item['hypothesis'])
        return x, item['label']
