
from dataclasses import dataclass

import pytest
import tensorflow as tf
import numpy as np

from ecoroar.tokenizer import SimpleTestTokenizer
from ecoroar.transform import SequenceIndentifier
from ecoroar.util import get_compiler

@pytest.fixture
def tokenizer():
    return SimpleTestTokenizer()


@pytest.fixture
def x(tokenizer):
    return tf.data.Dataset.from_tensor_slices([
        '[BOS] token token [EOS] [BOS] token token [EOS]',
        '[BOS] token token [BOS] token token [EOS] [PAD]',
        '[BOS] token [EOS] token [EOS] [PAD] [PAD] [PAD]',
        '[BOS] token [EOS] [EOS] [PAD] [PAD] [PAD] [PAD]',
        '[BOS] [EOS] [EOS] [PAD] [PAD] [PAD] [PAD] [PAD]',
    ]).map(lambda doc: tokenizer((doc, ))) \
      .batch(5) \
      .get_single_element()

@dataclass
class CompileConfig:
    name: str
    args: int

compile_configs = [
    CompileConfig('no_compile', { 'run_eagerly': True, 'jit_compile': False }),
    CompileConfig('default_compile', { 'run_eagerly': False, 'jit_compile': False }),
    CompileConfig('jit_compile', { 'run_eagerly': False, 'jit_compile': True })
]

@pytest.mark.parametrize("config", compile_configs, ids=lambda config: config.name)
def test_sequence_identifier(tokenizer, x, config):
    compiler = get_compiler(**config.args)
    identifier = compiler(SequenceIndentifier(tokenizer))
    sequence_idx = identifier(x).numpy()

    np.testing.assert_array_equal(sequence_idx, [
        [0, 1, 1, 0, 0, 2, 2, 0],
        [0, 1, 1, 0, 2, 2, 0, 0],
        [0, 1, 0, 2, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])
