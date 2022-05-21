
import pathlib
from ecoroar.dataset import datasets


def test_boolq_info():
    dataset = datasets['BoolQ'](persistent_dir=pathlib.Path('.'))

    assert dataset.name == 'BoolQ'
    assert dataset.train_num_examples == 9427
    assert dataset.valid_num_examples == 3270
    assert dataset.test_num_examples == 3245


def test_cola_info():
    dataset = datasets['CoLA'](persistent_dir=pathlib.Path('.'))

    assert dataset.name == 'CoLA'
    assert dataset.train_num_examples == 8551
    assert dataset.valid_num_examples == 1043
    assert dataset.test_num_examples == 1063


def test_imdb_info():
    dataset = datasets['IMDB'](persistent_dir=pathlib.Path('.'))

    assert dataset.name == 'IMDB'
    assert dataset.train_num_examples == 20000
    assert dataset.valid_num_examples == 5000
    assert dataset.test_num_examples == 25000


def test_mnli_info():
    dataset = datasets['MNLI'](persistent_dir=pathlib.Path('.'))

    assert dataset.name == 'MNLI'
    assert dataset.train_num_examples == 392702
    assert dataset.valid_num_examples == 9815
    assert dataset.test_num_examples == 9796


def test_qqp_info():
    dataset = datasets['QQP'](persistent_dir=pathlib.Path('.'))

    assert dataset.name == 'QQP'
    assert dataset.train_num_examples == 363846
    assert dataset.valid_num_examples == 40430
    assert dataset.test_num_examples == 390965


def test_sst2_info():
    dataset = datasets['SST2'](persistent_dir=pathlib.Path('.'))

    assert dataset.name == 'SST2'
    assert dataset.train_num_examples == 67349
    assert dataset.valid_num_examples == 872
    assert dataset.test_num_examples == 1821
