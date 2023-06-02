import tensorflow_datasets as tfds

from ._abstract_dataset import AbstractDataset


class MNLIDataset(AbstractDataset):
    _name = 'MNLI'
    _metrics = ['accuracy']
    _early_stopping_metric = 'accuracy'

    _split_train = 'train[:80%]'
    _split_valid = 'train[80%:]'
    _split_test = 'validation_matched'

    _class_count_train = [104768, 104689, 104705]
    _class_count_valid = [26131, 26211, 26198]
    _class_count_test = [3479, 3123, 3213]

    _input_masked = 'premise'
    _input_aux = 'hypothesis'

    def _builder(self, data_dir):
        return tfds.builder("glue/mnli", data_dir=data_dir)

    def _as_supervised(self, item):
        x = (item['premise'], item['hypothesis'])
        return x, item['label']
