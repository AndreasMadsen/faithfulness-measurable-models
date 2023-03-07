import tensorflow_datasets as tfds

from ._abstract_dataset import AbstractDataset
from .local import LocalMimic

class _MimicDatasetGeneralized(AbstractDataset):
    _metrics = ['accuracy', 'macro_f1']
    _early_stopping_metric = 'macro_f1'
    _target_name = 'diagnosis'

    _split_train = 'train'
    _split_valid = 'validation'
    _split_test = 'test'

    def _as_supervised(self, item):
        x = (item['text'], )
        return x, item['diagnosis']

class MimicAnemiaDataset(_MimicDatasetGeneralized):
    _name = 'MIMIC-a'

    def _builder(self, data_dir):
        return LocalMimic(data_dir=data_dir, config='anemia')

class MimicDiabetesDataset(_MimicDatasetGeneralized):
    _name = 'MIMIC-d'

    def _builder(self, data_dir):
        return LocalMimic(data_dir=data_dir, config='diabetes')
