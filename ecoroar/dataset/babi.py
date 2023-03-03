import tensorflow_datasets as tfds

from ._abstract_dataset import AbstractDataset
from .local import LocalBabi

class _BabiDatasetGeneralized(AbstractDataset):
    _metrics = ['accuracy']
    _early_stopping_metric = 'accuracy'

    _split_train = 'train[:80%]'
    _split_valid = 'train[80%:]'
    _split_test = 'test'

    def _as_supervised(self, item):
        x = (item['paragraph'], item['question'])
        return x, item['answer']

class Babi1Dataset(_BabiDatasetGeneralized):
    _name = 'bAbI-1'

    def _builder(self, data_dir):
        return LocalBabi(data_dir=data_dir, config='en-10k/qa1')

class Babi2Dataset(_BabiDatasetGeneralized):
    _name = 'bAbI-2'

    def _builder(self, data_dir):
        return LocalBabi(data_dir=data_dir, config='en-10k/qa2')

class Babi3Dataset(_BabiDatasetGeneralized):
    _name = 'bAbI-3'

    def _builder(self, data_dir):
        return LocalBabi(data_dir=data_dir, config='en-10k/qa3')
