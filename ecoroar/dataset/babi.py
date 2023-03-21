import tensorflow_datasets as tfds

from ._abstract_dataset import AbstractDataset
from .local import LocalBabi

class _BabiDatasetGeneralized(AbstractDataset):
    _metrics = ['accuracy', 'micro_f1']
    _early_stopping_metric = 'micro_f1'
    _target_name = 'answer'

    _split_train = 'train[:80%]'
    _split_valid = 'train[80%:]'
    _split_test = 'test'

    def _as_supervised(self, item):
        x = (item['paragraph'], item['question'])
        return x, item['answer']

class Babi1Dataset(_BabiDatasetGeneralized):
    _name = 'bAbI-1'
    _class_count_train = [1315, 1328, 1317, 1316, 1317, 1407]
    _class_count_valid = [348, 329, 327, 343, 327, 326]
    _class_count_test = [187, 154, 157, 182, 171, 149]

    def _builder(self, data_dir):
        return LocalBabi(data_dir=data_dir, config='en-10k/qa1')

class Babi2Dataset(_BabiDatasetGeneralized):
    _name = 'bAbI-2'
    _class_count_train = [1417, 1387, 1235, 1298, 1390, 1273]
    _class_count_valid = [353, 342, 315, 316, 365, 309]
    _class_count_test = [187, 160, 165, 146, 167, 175]

    def _builder(self, data_dir):
        return LocalBabi(data_dir=data_dir, config='en-10k/qa2')

class Babi3Dataset(_BabiDatasetGeneralized):
    _name = 'bAbI-3'
    _class_count_train = [1334, 1354, 1249, 1339, 1303, 1421]
    _class_count_valid = [352, 340, 324, 331, 308, 345]
    _class_count_test = [215, 167, 146, 133, 154, 185]

    def _builder(self, data_dir):
        return LocalBabi(data_dir=data_dir, config='en-10k/qa3')
