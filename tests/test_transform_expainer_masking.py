
from dataclasses import dataclass

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


@pytest.fixture
def dataset_paired(tokenizer):
    return tf.data.experimental.from_list([
        ('[BOS] token token [BOS] token [EOS] [PAD]', 0),
        ('[BOS] token [BOS] token [EOS] [PAD] [PAD]', 1)
    ]).map(lambda doc, y: (tokenizer((doc, )), y)) \
      .batch(2)


@pytest.fixture
def masked_dataset_paired(tokenizer):
    return tf.data.experimental.from_list([
        ('[BOS] token [MASK] [BOS] token [EOS] [PAD]', 0),
        ('[BOS] [MASK] [BOS] [MASK] [EOS] [PAD] [PAD]', 1)
    ]).map(lambda doc, y: (tokenizer((doc, )), y)) \
      .batch(2)

class ExplainerMaskingForcedIMCompute(ExplainerMasking):
    @tf.function(reduce_retracing=True)
    def _mask_input_100p(self, x, y):
        return self._mask_input(x, y, tf.constant(1.0, dtype=tf.dtypes.float32))

@dataclass
class ExplainerMaskingConfig:
    name: str
    ExplainerMasking: int

explainer_masking_configs = [
    ExplainerMaskingConfig('standard', ExplainerMasking),
    ExplainerMaskingConfig('mocked', ExplainerMaskingForcedIMCompute)
]

@pytest.mark.parametrize("config", explainer_masking_configs, ids=lambda config: config.name)
def test_explainer_masking_from_plain(tokenizer, model, dataset, config):
    explainer = GradientL2Explainer(tokenizer, model)
    masker = config.ExplainerMasking(explainer, tokenizer)

    x_masked_50, _ = dataset.apply(masker(0.5)).get_single_element()
    np.testing.assert_array_equal(x_masked_50['input_ids'], [
        [0, 4, 3, 1, 2],
        [0, 3, 1, 2, 2]
    ])
    np.testing.assert_array_equal(x_masked_50['attention_mask'], [
        [1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0]
    ])


@pytest.mark.parametrize("config", explainer_masking_configs, ids=lambda config: config.name)
def test_explainer_masking_from_masked(tokenizer, model, masked_dataset, config):
    explainer = GradientL2Explainer(tokenizer, model)
    masker = config.ExplainerMasking(explainer, tokenizer)

    x_masked_50, _ = masked_dataset.apply(masker(0.5)).get_single_element()
    np.testing.assert_array_equal(x_masked_50['input_ids'], [
        [0, 3, 4, 1, 2],
        [0, 4, 1, 2, 2]
    ])
    np.testing.assert_array_equal(x_masked_50['attention_mask'], [
        [1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0]
    ])


@pytest.mark.parametrize("config", explainer_masking_configs, ids=lambda config: config.name)
def test_explainer_masking_kept_tokens(tokenizer, model, dataset, config):
    explainer = GradientL2Explainer(tokenizer, model)
    masker = config.ExplainerMasking(explainer, tokenizer)

    x_masked_100, _ = dataset.apply(masker(1)).get_single_element()
    np.testing.assert_array_equal(x_masked_100['input_ids'], [
        [0, 4, 4, 1, 2],
        [0, 4, 1, 2, 2]
    ])
    np.testing.assert_array_equal(x_masked_100['attention_mask'], [
        [1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0]
    ])


@pytest.mark.parametrize("config", explainer_masking_configs, ids=lambda config: config.name)
def test_explainer_masking_from_paired(tokenizer, model, dataset_paired, config):
    explainer = GradientL2Explainer(tokenizer, model)
    masker = config.ExplainerMasking(explainer, tokenizer)

    x_masked_50, _ = dataset_paired.apply(masker(0.5)).get_single_element()
    np.testing.assert_array_equal(x_masked_50['input_ids'], [
        [0, 4, 3, 0, 3, 1, 2],
        [0, 3, 0, 3, 1, 2, 2]
    ])
    np.testing.assert_array_equal(x_masked_50['attention_mask'], [
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 0, 0]
    ])

    x_masked_100, _ = dataset_paired.apply(masker(1)).get_single_element()
    np.testing.assert_array_equal(x_masked_100['input_ids'], [
        [0, 4, 4, 0, 3, 1, 2],
        [0, 4, 0, 3, 1, 2, 2]
    ])
    np.testing.assert_array_equal(x_masked_100['attention_mask'], [
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 0, 0]
    ])


@pytest.mark.parametrize("config", explainer_masking_configs, ids=lambda config: config.name)
def test_explainer_masking_from_masked_paired(tokenizer, model, masked_dataset_paired, config):
    explainer = GradientL2Explainer(tokenizer, model)
    masker = config.ExplainerMasking(explainer, tokenizer)

    x_masked_50, _ = masked_dataset_paired.apply(masker(0.5)).get_single_element()
    np.testing.assert_array_equal(x_masked_50['input_ids'], [
        [0, 3, 4, 0, 3, 1, 2],
        [0, 4, 0, 4, 1, 2, 2]
    ])
    np.testing.assert_array_equal(x_masked_50['attention_mask'], [
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 0, 0]
    ])

    x_masked_100, _ = masked_dataset_paired.apply(masker(1)).get_single_element()
    np.testing.assert_array_equal(x_masked_100['input_ids'], [
        [0, 4, 4, 0, 3, 1, 2],
        [0, 4, 0, 4, 1, 2, 2]
    ])
    np.testing.assert_array_equal(x_masked_100['attention_mask'], [
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 0, 0]
    ])
