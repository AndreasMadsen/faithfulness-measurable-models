import tensorflow_datasets as tfds

from ._abstract_dataset import AbstractDataset


class MNLIDataset(AbstractDataset):
    _name = 'MNLI'
    _metric = 'accuracy'

    _split_train = 'train'
    _split_valid = 'validation_matched'
    _split_test = 'test_matched'

    def _builder(self, data_dir):
        return tfds.builder("glue/mnli", data_dir=data_dir)

    def _as_supervised(self, item):
        x = (item['premise'], item['hypothesis'])
        return x, item['label']
