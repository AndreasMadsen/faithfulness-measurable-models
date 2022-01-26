import pytest

from ecoroar.dataset import IMDBDataset

def test_imdb_info():
    dataset_tf = IMDBDataset(persistent_dir='.')

    assert dataset_tf.train_num_examples == 20000
    assert dataset_tf.valid_num_examples == 5000
    assert dataset_tf.test_num_examples == 25000
