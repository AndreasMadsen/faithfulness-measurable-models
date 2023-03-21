import tensorflow_datasets as tfds

from ._abstract_dataset import AbstractDataset


class QNLIDataset(AbstractDataset):
    _name = 'QNLI'
    _metrics = ['accuracy']
    _early_stopping_metric = 'accuracy'

    _split_train = 'train[:80%]'
    _split_valid = 'train[80%:]'
    _split_test = 'validation'

    _class_count_train = [41889, 41905]
    _class_count_valid = [10488, 10461]
    _class_count_test = [2702, 2761]

    def _builder(self, data_dir):
        return tfds.builder("glue/qnli", data_dir=data_dir)

    def _as_supervised(self, item):
        x = (item['sentence'], item['question'])
        return x, item['label']
