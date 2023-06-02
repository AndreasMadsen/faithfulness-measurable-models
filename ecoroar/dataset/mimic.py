import tensorflow_datasets as tfds

from ._abstract_dataset import AbstractDataset
from .local import LocalMimic

class _MimicDatasetGeneralized(AbstractDataset):
    _metrics = ['accuracy', 'macro_f1']
    _early_stopping_metric = 'macro_f1'
    _jain_etal_metric = 'macro_f1'
    _target_name = 'diagnosis'

    _split_train = 'train'
    _split_valid = 'validation'
    _split_test = 'test'

    _input_masked = 'text'

    def _as_supervised(self, item):
        x = (item['text'], )
        return x, item['diagnosis']

class MimicAnemiaDataset(_MimicDatasetGeneralized):
    _name = 'MIMIC-a'

    _class_count_train = [1522, 2740]
    _class_count_valid = [263, 466]
    _class_count_test = [450, 793]

    def _builder(self, data_dir):
        return LocalMimic(data_dir=data_dir, config='anemia')

class MimicDiabetesDataset(_MimicDatasetGeneralized):
    _name = 'MIMIC-d'

    _class_count_train = [6650, 1416]
    _class_count_valid = [1284, 289]
    _class_count_test = [1389, 340]

    def _builder(self, data_dir):
        return LocalMimic(data_dir=data_dir, config='diabetes')
