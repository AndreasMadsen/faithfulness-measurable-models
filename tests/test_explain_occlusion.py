
import pytest
import tensorflow as tf
import numpy as np

from ecoroar.model import LookupTestModel
from ecoroar.tokenizer import SimpleTestTokenizer
from ecoroar.explain import LeaveOneOutAbs, LeaveOneOutSign, BeamSearch
from ecoroar.test import compile_configs


@pytest.fixture
def tokenizer():
    return SimpleTestTokenizer()

@pytest.fixture
def model(tokenizer):
    return LookupTestModel.from_string(tokenizer, mapping={
        '[BOS] token  [EOS]  [PAD]': (-2.0, 2.0),
        '[BOS] [MASK] [EOS]  [PAD]': (-3.0, 3.0),

        '[BOS] token  token  [EOS]': (2.0, -2.0),
        '[BOS] token  [MASK] [EOS]': (1.0, -1.0),
        '[BOS] [MASK] token  [EOS]': (-1.0, 1.0),
        '[BOS] [MASK] [MASK] [EOS]': (1.0, -1.0),

        # B-1: 3      1      4      2
        # B-2: 4      3      2      1
        # B-3: 1      3      2      4
        # B-4: 1      3      2      4
        '[BOS] token  token  token  token  [EOS]': (5.0, 0.0),
        '[BOS] [MASK] [MASK] [MASK] [MASK] [EOS]': (2.0, 0.0),

        #      [MASK]
        '[BOS] [MASK] token  token  token  [EOS]': (2.0, 0.0), # 1:3
        #      [MASK] [MASK]
        '[BOS] [MASK] [MASK] token  token  [EOS]': (3.0, 0.0), # 12:5*, 21:-2
        '[BOS] [MASK] [MASK] [MASK] token  [EOS]': (3.0, 0.0), # 123:7, 312:5
        '[BOS] [MASK] [MASK] token  [MASK] [EOS]': (5.0, 0.0), # 124:5, 421:3
        #      [MASK]        [MASK]
        '[BOS] [MASK] token  [MASK] token  [EOS]': (6.0, 0.0), # 31:3*, 13:2*
        '[BOS] [MASK] token  [MASK] [MASK] [EOS]': (2.0, 0.0), # 314:6
        #      [MASK]               [MASK]
        '[BOS] [MASK] token  token  [MASK] [EOS]': (7.0, 0.0), # 14:1, 41:-2

        #             [MASK]
        '[BOS] token  [MASK] token  token  [EOS]': (9.0, 0.0), # 2:-4
        #             [MASK] [MASK]
        '[BOS] token  [MASK] [MASK] token  [EOS]': (8.0, 0.0), # 32:1, 23:-7
        '[BOS] token  [MASK] [MASK] [MASK] [EOS]': (0.0, 0.0), # 423:8
        #             [MASK]        [MASK]
        '[BOS] token  [MASK] token  [MASK] [EOS]': (2.0, 0.0), # 42:3*, 24:-1

        #                    [MASK]
        '[BOS] token  token  [MASK] token  [EOS]': (1.0, 0.0), # 3:4
        #                    [MASK] [MASK]
        '[BOS] token  token  [MASK] [MASK] [EOS]': (8.0, 0.0), # 34:1, 42:-3

        #                           [MASK]
        '[BOS] token  token  token  [MASK] [EOS]': (5.0, 0.0), # 4:0
    })

@pytest.fixture
def x_12(tokenizer):
    return tf.data.Dataset.from_tensor_slices([
        '[BOS] token token [EOS]',
        '[BOS] token [EOS] [PAD]',
    ]).map(lambda doc: tokenizer((doc, ))) \
      .batch(2) \
      .get_single_element()

@pytest.fixture
def x_4(tokenizer):
    return tf.data.Dataset.from_tensor_slices([
        '[BOS] token token token token [EOS]',
    ]).map(lambda doc: tokenizer((doc, ))) \
      .batch(1) \
      .get_single_element()


@pytest.mark.parametrize("config", compile_configs, ids=lambda config: config.name)
def test_explainer_leave_on_out_abs(tokenizer, model, x_12, config):
    explainer = LeaveOneOutAbs(tokenizer, model, **config.args)

    im = explainer(x_12, tf.constant([0, 1])).to_tensor(default_value=-1).numpy()
    np.testing.assert_allclose(im, [
        [0, 3, 1, 0],
        [0, 1, 0, -1]
    ])

@pytest.mark.parametrize("config", compile_configs, ids=lambda config: config.name)
def test_explainer_leave_on_out_sign(tokenizer, model, x_12, config):
    explainer = LeaveOneOutSign(tokenizer, model, **config.args)

    im = explainer(x_12, tf.constant([0, 1])).to_tensor(default_value=-1).numpy()
    np.testing.assert_allclose(im, [
        [0, 3, 1, 0],
        [0, -1, 0, -1]
    ])

@pytest.mark.parametrize("config", compile_configs, ids=lambda config: config.name)
def test_explainer_beam_search_size_1(tokenizer, model, x_4, config):
    explainer = BeamSearch(tokenizer, model, beam_size=1, **config.args)

    im = explainer(x_4, tf.constant([0])).to_tensor(default_value=-1).numpy()
    np.testing.assert_allclose(im, [
        [0, 3, 1, 4, 2, 0],
    ])

@pytest.mark.parametrize("config", compile_configs, ids=lambda config: config.name)
def test_explainer_beam_search_size_2(tokenizer, model, x_4, config):
    explainer = BeamSearch(tokenizer, model, beam_size=2, **config.args)

    im = explainer(x_4, tf.constant([0])).to_tensor(default_value=-1).numpy()
    np.testing.assert_allclose(im, [
        [0, 4, 3, 2, 1, 0],
    ])

@pytest.mark.parametrize("config", compile_configs, ids=lambda config: config.name)
@pytest.mark.parametrize("beam_size", list(range(3, 10)))
def test_explainer_beam_search_size_above_2(tokenizer, model, x_4, config, beam_size):
    explainer = BeamSearch(tokenizer, model, beam_size=beam_size, **config.args)

    im = explainer(x_4, tf.constant([0])).to_tensor(default_value=-1).numpy()
    np.testing.assert_allclose(im, [
        [0, 1, 3, 2, 4, 0],
    ])
