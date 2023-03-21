import tensorflow_datasets as tfds

from ._abstract_dataset import AbstractDataset


class MRPCDataset(AbstractDataset):
    _name = 'MRPC'
    _metrics = ['accuracy', 'macro_f1']
    _early_stopping_metric = 'macro_f1'

    _split_train = 'train[:80%]'
    _split_valid = 'train[80%:]'
    _split_test = 'validation'

    _class_count_train = [961, 1973]
    _class_count_valid = [233, 501]
    _class_count_test = [129, 279]

    def _builder(self, data_dir):
        return tfds.builder("glue/mrpc", data_dir=data_dir)

    def _as_supervised(self, item):
        x = (item['sentence1'], item['sentence2'])
        return x, item['label']
