import tensorflow_datasets as tfds

from ._abstract_dataset import AbstractDataset
from .local import LocalSNLI

class SNLIDataset(AbstractDataset):
    _name = 'SNLI'
    _metrics = ['accuracy', 'macro_f1']
    _early_stopping_metric = 'accuracy'
    _jain_etal_metric = 'macro_f1'

    _split_train = 'train'
    _split_valid = 'validation'
    _split_test = 'test'

    _class_count_train = [183416, 182764, 183187]
    _class_count_valid = [3329, 3235, 3278]
    _class_count_test = [3368, 3219, 3237]

    def _builder(self, data_dir):
        return LocalSNLI(data_dir=data_dir)

    def _as_supervised(self, item):
        x = (item['premise'], item['hypothesis'])
        return x, item['label']
