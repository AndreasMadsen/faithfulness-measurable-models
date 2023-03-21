import tensorflow_datasets as tfds

from ._abstract_dataset import AbstractDataset


class BoolQDataset(AbstractDataset):
    _name = 'BoolQ'
    _metrics = ['accuracy']
    _early_stopping_metric = 'accuracy'

    _split_train = 'train[:80%]'
    _split_valid = 'train[80%:]'
    _split_test = 'validation'

    _class_count_train = [2809, 4733]
    _class_count_valid = [744, 1141]
    _class_count_test = [1237, 2033]

    def _builder(self, data_dir):
        return tfds.builder("super_glue/boolq", data_dir=data_dir)

    def _as_supervised(self, item):
        x = (item['question'], item['passage'])
        return x, item['label']
