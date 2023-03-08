import tensorflow_datasets as tfds

from ._abstract_dataset import AbstractDataset


class SST2Dataset(AbstractDataset):
    _name = 'SST2'
    _metrics = ['accuracy']
    _early_stopping_metric = 'accuracy'

    _split_train = 'train[:80%]'
    _split_valid = 'train[80%:]'
    _split_test = 'validation'

    _class_count_train = [23825, 30054]
    _class_count_valid = [5955, 7515]
    _class_count_test = [428, 444]

    def _builder(self, data_dir):
        return tfds.builder("glue/sst2", data_dir=data_dir)

    def _as_supervised(self, item):
        x = (item['sentence'], )
        return x, item['label']
