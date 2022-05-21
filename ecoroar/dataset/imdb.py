import tensorflow_datasets as tfds

from ._abstract_dataset import AbstractDataset


class IMDBDataset(AbstractDataset):
    _name = 'IMDB'
    _metrics = ['accuracy', 'macro_f1']
    _early_stopping_metric = 'macro_f1'

    _split_train = 'train[:80%]'
    _split_valid = 'train[80%:]'
    _split_test = 'test'

    def _builder(self, data_dir):
        return tfds.builder("imdb_reviews", data_dir=data_dir)

    def _as_supervised(self, item):
        x = (item['text'], )
        return x, item['label']
