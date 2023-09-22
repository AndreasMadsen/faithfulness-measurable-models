
from dataclasses import dataclass
import pytest
import tensorflow as tf
import numpy as np

from ecoroar.model import SimpleTestModel
from ecoroar.tokenizer import SimpleTestTokenizer
from ecoroar.ood import ood_detectors
from ecoroar.test import compile_configs


@pytest.fixture
def tokenizer():
    return SimpleTestTokenizer()


@pytest.fixture
def model():
    return SimpleTestModel()


@pytest.fixture
def dataset(tokenizer):
    return tf.data.Dataset.from_tensor_slices([
        '[BOS] token token token token token [EOS] [PAD]',
        '[BOS] token token token token [EOS] [PAD] [PAD]',
        '[BOS] token token token [EOS] [PAD]',
        '[BOS] token token [EOS] [PAD] [PAD]',
        '[BOS] token [EOS] [PAD]',
        '[BOS] [EOS] [PAD] [PAD]',
    ]).map(lambda doc: (tokenizer((doc, )), 0)) \
      .batch(2)


@pytest.mark.parametrize("compile_config", compile_configs, ids=lambda config: config.name)
@pytest.mark.parametrize("ood_detector", ood_detectors.values(), ids=lambda ood_detector: ood_detector._name)
def test_odd_mafs(tokenizer, model, dataset, ood_detector, compile_config):
    dist = ood_detector(tokenizer, model, verbose=False, **compile_config.args)
    dist.fit(dataset)

    # Note that due to the simplicity of the SimpleTestModel, the intermediate
    # representations are too sparse and discrete to provide a satifiying
    # distribution. So everything, is more or less going to test as
    # in-distribution.
    for ood in dataset.apply(dist).rebatch(2):
        np.testing.assert_allclose(ood.numpy(), [1, 1])
