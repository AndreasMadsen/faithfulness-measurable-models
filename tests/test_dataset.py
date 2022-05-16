
import pathlib
from ecoroar.dataset import datasets


def test_imdb_info():
    dataset = datasets['IMDB'](persistent_dir=pathlib.Path('.'))

    assert dataset.name == 'IMDB'
    assert dataset.train_num_examples == 20000
    assert dataset.valid_num_examples == 5000
    assert dataset.test_num_examples == 25000


def test_multi_nli_info():
    dataset = datasets['MultiNLI'](persistent_dir=pathlib.Path('.'))

    assert dataset.name == 'MultiNLI'
    assert dataset.train_num_examples == 314162
    assert dataset.valid_num_examples == 78540
    assert dataset.test_num_examples == 9815
