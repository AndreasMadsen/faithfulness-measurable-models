import tensorflow_datasets as tfds

from ._abstract_dataset import AbstractDataset


class BoolQDataset(AbstractDataset):
    _name = 'BoolQ'
    _metrics = ['accuracy']
    _early_stopping_metric = 'accuracy'

    _split_train = 'train'
    _split_valid = 'validation'
    _split_test = 'test'

    def _builder(self, data_dir):
        return tfds.builder("super_glue/boolq", data_dir=data_dir)

    def _as_supervised(self, item):
        x = (item['question'], item['passage'])
        return x, item['label']
