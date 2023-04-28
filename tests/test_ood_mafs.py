
from dataclasses import dataclass
import pytest
import tensorflow as tf
import numpy as np

from ecoroar.model import SimpleTestModel
from ecoroar.tokenizer import SimpleTestTokenizer
from ecoroar.ood import MaSF

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

@dataclass
class CompileConfig:
    name: str
    args: int

compile_configs = [
    CompileConfig('no_compile', { 'run_eagerly': True, 'jit_compile': False }),
    CompileConfig('default_compile', { 'run_eagerly': False, 'jit_compile': False }),
    CompileConfig('jit_compile', { 'run_eagerly': False, 'jit_compile': True })
]

@pytest.mark.parametrize("compile_config", compile_configs, ids=lambda config: config.name)
def test_odd_mafs(tokenizer, model, dataset, compile_config):
    dist = MaSF(tokenizer, model, verbose=False, **compile_config.args)
    dist.fit(dataset)

    # Note that due to the simplicity of the SimpleTestModel, the intermediate
    # representations are too sparse and discrete to provide a satifiying
    # distribution. So everything, is more or less going to test as
    # in-distribution.
    for x, _ in dataset:
        np.testing.assert_allclose(dist(x).numpy(), [1, 1])
