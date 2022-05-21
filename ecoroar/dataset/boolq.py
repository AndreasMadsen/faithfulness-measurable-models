import tensorflow_datasets as tfds

from ._abstract_dataset import AbstractDataset


class BoolQDataset(AbstractDataset):
    _name = 'BoolQ'
    _metrics = ['accuracy']
    _early_stopping_metric = 'accuracy'

    _split_train = 'train[:80%]'
    _split_valid = 'train[80%:]'
    _split_test = 'validation'

    def _builder(self, data_dir):
        return tfds.builder("super_glue/boolq", data_dir=data_dir)

    def _as_supervised(self, item):
        x = (item['question'], item['passage'])
        return x, item['label']
