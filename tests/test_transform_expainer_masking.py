
import pytest
import tensorflow as tf
import numpy as np

from ecoroar.model import SimpleTestModel
from ecoroar.tokenizer import SimpleTestTokenizer
from ecoroar.explain import GradientL2Explainer
from ecoroar.transform import ExplainerMasking


@pytest.fixture
def tokenizer():
    return SimpleTestTokenizer()


@pytest.fixture
def model():
    return SimpleTestModel()


@pytest.fixture
def dataset(tokenizer):
    return tf.data.experimental.from_list([
        ('[BOS] token token [EOS] [PAD]', 0),
        ('[BOS] token [EOS] [PAD] [PAD]', 1)
    ]).map(lambda doc, y: (tokenizer((doc, )), y)) \
      .batch(2)


@pytest.fixture
def masked_dataset(tokenizer):
    return tf.data.experimental.from_list([
        ('[BOS] token [MASK] [EOS] [PAD]', 0),
        ('[BOS] [MASK] [EOS] [PAD] [PAD]', 1)
    ]).map(lambda doc, y: (tokenizer((doc, )), y)) \
      .batch(2)


def test_explainer_masking_from_plain(tokenizer, model, dataset):
    explainer = GradientL2Explainer(tokenizer, model)
    masker = ExplainerMasking(explainer, tokenizer)

    x_masked_50, _ = dataset.apply(masker(0.5)).get_single_element()
    np.testing.assert_array_equal(x_masked_50['input_ids'], [
        [0, 4, 3, 1, 2],
        [0, 3, 1, 2, 2]
    ])
    np.testing.assert_array_equal(x_masked_50['attention_mask'], [
        [1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0]
    ])


def test_explainer_masking_from_masked(tokenizer, model, masked_dataset):
    explainer = GradientL2Explainer(tokenizer, model)
    masker = ExplainerMasking(explainer, tokenizer)

    x_masked_50, _ = masked_dataset.apply(masker(0.5)).get_single_element()
    np.testing.assert_array_equal(x_masked_50['input_ids'], [
        [0, 3, 4, 1, 2],
        [0, 4, 1, 2, 2]
    ])
    np.testing.assert_array_equal(x_masked_50['attention_mask'], [
        [1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0]
    ])


def test_explainer_masking_kept_tokens(tokenizer, model, dataset):
    explainer = GradientL2Explainer(tokenizer, model)
    masker = ExplainerMasking(explainer, tokenizer)

    x_masked_100, _ = dataset.apply(masker(1)).get_single_element()
    np.testing.assert_array_equal(x_masked_100['input_ids'], [
        [0, 4, 4, 1, 2],
        [0, 4, 1, 2, 2]
    ])
    np.testing.assert_array_equal(x_masked_100['attention_mask'], [
        [1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0]
    ])
