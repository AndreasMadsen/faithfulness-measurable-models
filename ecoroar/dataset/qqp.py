import tensorflow_datasets as tfds

from ._abstract_dataset import AbstractDataset


class QQPDataset(AbstractDataset):
    _name = 'QQP'
    _metric = 'macro_f1'

    _split_train = 'train'
    _split_valid = 'validation'
    _split_test = 'test'

    def _builder(self, data_dir):
        return tfds.builder("glue/qqp", data_dir=data_dir)

    def _as_supervised(self, item):
        x = (item['question1'], item['question2'])
        return x, item['label']
