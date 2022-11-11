import tensorflow_datasets as tfds

from ._abstract_dataset import AbstractDataset


class MNLIDataset(AbstractDataset):
    _name = 'MNLI'
    _metrics = ['accuracy']
    _early_stopping_metric = 'accuracy'

    _split_train = 'train[:80%]'
    _split_valid = 'train[80%:]'
    _split_test = 'validation_matched'

    def _builder(self, data_dir):
        return tfds.builder("glue/mnli", data_dir=data_dir)

    def _as_supervised(self, item):
        x = (item['premise'], item['hypothesis'])
        return x, item['label']
