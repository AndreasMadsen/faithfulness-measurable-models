import tensorflow_datasets as tfds

from ._abstract_dataset import AbstractDataset


class WNLIDataset(AbstractDataset):
    _name = 'WNLI'
    _metrics = ['accuracy']
    _early_stopping_metric = 'accuracy'

    _split_train = 'train[:80%]'
    _split_valid = 'train[80%:]'
    _split_test = 'validation'

    _class_count_train = [250, 258]
    _class_count_valid = [73, 54]
    _class_count_test = [40, 31]

    _input_masked = 'sentence1'
    _input_aux = 'sentence2'

    def _builder(self, data_dir):
        return tfds.builder("glue/wnli", data_dir=data_dir)

    def _as_supervised(self, item):
        x = (item['sentence1'], item['sentence2'])
        return x, item['label']
