import tensorflow_datasets as tfds

from ._abstract_dataset import AbstractDataset


class SST2Dataset(AbstractDataset):
    _name = 'SST2'
    _metric = 'pearson'

    _split_train = 'train'
    _split_valid = 'validation'
    _split_test = 'test'

    def _builder(self, data_dir):
        return tfds.builder("glue/sst2", data_dir=data_dir)

    def _as_supervised(self, item):
        x = (item['sentence'], )
        return x, item['label']
