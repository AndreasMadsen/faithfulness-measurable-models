import tensorflow_datasets as tfds

from ._abstract_dataset import AbstractDataset


class CoLADataset(AbstractDataset):
    _name = 'CoLA'
    _metrics = ['accuracy', 'matthew']
    _early_stopping_metric = 'matthew'

    _split_train = 'train[:80%]'
    _split_valid = 'train[80%:]'
    _split_test = 'validation'

    _class_count_train = [2014, 4827]
    _class_count_valid = [514, 1196]
    _class_count_test = [322, 721]

    _input_masked = 'sentence'

    def _builder(self, data_dir):
        return tfds.builder("glue/cola", data_dir=data_dir)

    def _as_supervised(self, item):
        x = (item['sentence'], )
        return x, item['label']
