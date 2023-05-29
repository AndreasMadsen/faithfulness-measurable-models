
from dataclasses import dataclass

import pytest
import tensorflow as tf
import numpy as np

from ecoroar.tokenizer import SimpleTestTokenizer
from ecoroar.transform import SequenceIndentifier
from ecoroar.util import get_compiler
from ecoroar.test import compile_configs

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

@pytest.mark.parametrize("config", compile_configs, ids=lambda config: config.name)
def test_sequence_identifier(tokenizer, x, config):
    compiler = get_compiler(**config.args)
    identifier = compiler(SequenceIndentifier(tokenizer))
    sequence_idx = identifier(x['input_ids']).numpy()

    np.testing.assert_array_equal(sequence_idx, [
        [0, 1, 1, 0, 0, 2, 2, 0],
        [0, 1, 1, 0, 2, 2, 0, 0],
        [0, 1, 0, 2, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])
