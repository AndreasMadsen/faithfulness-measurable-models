
from dataclasses import dataclass

import pytest
import tensorflow as tf
import numpy as np

from ecoroar.model import LookupTestModel
from ecoroar.tokenizer import SimpleTestTokenizer
from ecoroar.explain import LeaveOneOutAbs, LeaveOneOutSign
from ecoroar.test import compile_configs


@pytest.fixture
def tokenizer():
    return SimpleTestTokenizer()


@pytest.fixture
def model(tokenizer):
    return LookupTestModel.from_string(tokenizer, mapping={
        '[BOS] token  token  [EOS]': (2.0, -2.0),
        '[BOS] token  [MASK] [EOS]': (1.0, -1.0),
        '[BOS] [MASK] token  [EOS]': (-1.0, 1.0),
        '[BOS] [MASK] [MASK] [EOS]': (1.0, -1.0),
        '[BOS] token  [EOS]  [PAD]': (-2.0, 2.0),
        '[BOS] [MASK] [EOS]  [PAD]': (-3.0, 3.0),
    })


@pytest.fixture
def x(tokenizer):
    return tf.data.Dataset.from_tensor_slices([
        '[BOS] token token [EOS]',
        '[BOS] token [EOS] [PAD]',
    ]).map(lambda doc: tokenizer((doc, ))) \
      .batch(2) \
      .get_single_element()


#@pytest.mark.parametrize("config", compile_configs, ids=lambda config: config.name)
def test_explainer_leave_on_out_abs(tokenizer, model, x):
    explainer = LeaveOneOutAbs(tokenizer, model)

    im = explainer(x, tf.constant([0, 1])).to_tensor(default_value=-1).numpy()
    np.testing.assert_allclose(im, [
        [0, 3, 1, 0],
        [0, 1, 0, -1]
    ])

#@pytest.mark.parametrize("config", compile_configs, ids=lambda config: config.name)
def test_explainer_leave_on_out_sign(tokenizer, model, x):
    explainer = LeaveOneOutSign(tokenizer, model)

    im = explainer(x, tf.constant([0, 1])).to_tensor(default_value=-1).numpy()
    np.testing.assert_allclose(im, [
        [0, 3, 1, 0],
        [0, -1, 0, -1]
    ])
