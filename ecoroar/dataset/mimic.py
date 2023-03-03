import tensorflow_datasets as tfds

from ._abstract_dataset import AbstractDataset
from .local import LocalMimic

class _BabiDatasetGeneralized(AbstractDataset):
    _metrics = ['accuracy', 'macro-f1']
    _early_stopping_metric = 'macro-f1'

    _split_train = 'train'
    _split_valid = 'validation'
    _split_test = 'test'

    def _as_supervised(self, item):
        x = (item['text'], )
        return x, item['diagnosis']

class AnemiaDataset(_BabiDatasetGeneralized):
    _name = 'Anemia'

    def _builder(self, data_dir):
        return LocalMimic(data_dir=data_dir, config='anemia')

class DiabetesDataset(_BabiDatasetGeneralized):
    _name = 'Diabetes'

    def _builder(self, data_dir):
        return LocalMimic(data_dir=data_dir, config='diabetes')
