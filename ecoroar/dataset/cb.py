import tensorflow_datasets as tfds

from ._abstract_dataset import AbstractDataset


class CBDataset(AbstractDataset):
    _name = 'CB'
    _metrics = ['accuracy', 'macro_f1']
    _early_stopping_metric = 'macro_f1'

    _split_train = 'train[:80%]'
    _split_valid = 'train[80%:]'
    _split_test = 'validation'

    def _builder(self, data_dir):
        return tfds.builder("super_glue/cb", data_dir=data_dir)

    def _as_supervised(self, item):
        x = (item['premise'], item['hypothesis'])
        return x, item['label']
