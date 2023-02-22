
import pytest
import tensorflow as tf
import numpy as np

from ecoroar.model import SimpleTestModel
from ecoroar.tokenizer import SimpleTestTokenizer
from ecoroar.explain import GradientExplainer, InputTimesGradientExplainer


@pytest.fixture
def tokenizer():
    return SimpleTestTokenizer()


@pytest.fixture
def model():
    return SimpleTestModel()


@pytest.fixture
def x(tokenizer):
    return tf.data.Dataset.from_tensor_slices([
        '[BOS] token token [EOS] [PAD]',
        '[BOS] token [EOS] [PAD] [PAD]',
    ]).map(lambda doc: tokenizer((doc, ))) \
      .batch(2) \
      .get_single_element()


def test_explainer_gradient(tokenizer, model, x):
    explainer = GradientExplainer(tokenizer, model)

    im = (explainer(x, tf.constant([0, 1])) ** 2).to_tensor(default_value=-1).numpy()
    np.testing.assert_allclose(im, [
        [0,  4,  4, 0],
        [12, 0, 12, -1]
    ])

def test_explainer_input_times_gradient(tokenizer, model, x):
    explainer = InputTimesGradientExplainer(tokenizer, model)

    im = (explainer(x, tf.constant([0, 1])) ** 2).to_tensor(default_value=-1).numpy()
    np.testing.assert_allclose(im, [
        [0, 4, 4, 0],
        [4, 0, 4, -1]
    ])
