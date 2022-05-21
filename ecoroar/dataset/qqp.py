import tensorflow_datasets as tfds

from ._abstract_dataset import AbstractDataset


class QQPDataset(AbstractDataset):
    _name = 'QQP'
    _metrics = ['accuracy', 'macro_f1']
    _early_stopping_metric = 'macro_f1'

    _split_train = 'train[:80%]'
    _split_valid = 'train[80%:]'
    _split_test = 'validation'

    def _builder(self, data_dir):
        return tfds.builder("glue/qqp", data_dir=data_dir)

    def _as_supervised(self, item):
        x = (item['question1'], item['question2'])
        return x, item['label']
