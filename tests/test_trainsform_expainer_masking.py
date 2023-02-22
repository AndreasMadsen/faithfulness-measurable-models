
import pytest
import tensorflow as tf
import numpy as np

from ecoroar.model import SimpleTestModel
from ecoroar.tokenizer import SimpleTestTokenizer
from ecoroar.explain import GradientExplainer
from ecoroar.transform import ExplainerMasking


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


@pytest.fixture
def x_masked(tokenizer):
    return tf.data.Dataset.from_tensor_slices([
        '[BOS] token [MASK] [EOS] [PAD]',
        '[BOS] [MASK] [EOS] [PAD] [PAD]',
    ]).map(lambda doc: tokenizer((doc, ))) \
      .batch(2) \
      .get_single_element()


def test_explainer_masking_from_plain(tokenizer, model, x):
    explainer = GradientExplainer(tokenizer, model)

    masker_50 = ExplainerMasking(0.5, explainer, tokenizer)
    x_masked_50 = masker_50(x, tf.constant([0, 1]))
    np.testing.assert_array_equal(x_masked_50['input_ids'], [
        [0, 4, 3, 1, 2],
        [0, 3, 1, 2, 2]
    ])
    np.testing.assert_array_equal(x_masked_50['attention_mask'], [
        [1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0]
    ])


def test_explainer_masking_from_masked(tokenizer, model, x_masked):
    explainer = GradientExplainer(tokenizer, model)

    masker_50 = ExplainerMasking(0.5, explainer, tokenizer)
    x_masked_50 = masker_50(x_masked, tf.constant([0, 1]))
    np.testing.assert_array_equal(x_masked_50['input_ids'], [
        [0, 3, 4, 1, 2],
        [0, 4, 1, 2, 2]
    ])
    np.testing.assert_array_equal(x_masked_50['attention_mask'], [
        [1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0]
    ])


def test_explainer_masking_kept_tokens(tokenizer, model, x):
    explainer = GradientExplainer(tokenizer, model)

    masker_100 = ExplainerMasking(1, explainer, tokenizer)
    x_masked_100 = masker_100(x, tf.constant([0, 1]))
    np.testing.assert_array_equal(x_masked_100['input_ids'], [
        [0, 4, 4, 1, 2],
        [0, 4, 1, 2, 2]
    ])
    np.testing.assert_array_equal(x_masked_100['attention_mask'], [
        [1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0]
    ])
