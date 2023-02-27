
from dataclasses import dataclass

import pytest
import tensorflow as tf
import numpy as np

from ecoroar.model import SimpleTestModel
from ecoroar.tokenizer import SimpleTestTokenizer
from ecoroar.explain import \
    RandomExplainer, GradientExplainer, InputTimesGradientExplainer, IntegratedGradientExplainer


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

@dataclass
class CompileConfig:
    name: str
    args: int

compile_configs = [
    CompileConfig('no_compile', { 'run_eagerly': True, 'jit_compile': False }),
    CompileConfig('default_compile', { 'run_eagerly': False, 'jit_compile': False }),
    CompileConfig('jit_compile', { 'run_eagerly': False, 'jit_compile': True })
]

def test_explainer_random(tokenizer, model, x):
    explainer = RandomExplainer(tokenizer, model, seed=0)

    im = (explainer(x, tf.constant([0, 1])) ** 2).to_tensor(default_value=-1).numpy()
    np.testing.assert_allclose(im, [
        [9.7e-02, 6.8e-01, 4.7e-01, 4.5e-05],
        [1.5e-01, 8.6e-02, 9.8e-01, -1]
    ], rtol=0.1)

@pytest.mark.parametrize("config", compile_configs, ids=lambda config: config.name)
def test_explainer_gradient(tokenizer, model, x, config):
    explainer = GradientExplainer(tokenizer, model, **config.args)

    im = (explainer(x, tf.constant([0, 1])) ** 2).to_tensor(default_value=-1).numpy()
    np.testing.assert_allclose(im, [
        [0,  4,  4, 0],
        [12, 0, 12, -1]
    ])

@pytest.mark.parametrize("config", compile_configs, ids=lambda config: config.name)
def test_explainer_input_times_gradient(tokenizer, model, x, config):
    explainer = InputTimesGradientExplainer(tokenizer, model, **config.args)

    im = (explainer(x, tf.constant([0, 1])) ** 2).to_tensor(default_value=-1).numpy()
    np.testing.assert_allclose(im, [
        [0, 4, 4, 0],
        [4, 0, 4, -1]
    ])

@pytest.mark.parametrize("config", compile_configs, ids=lambda config: config.name)
def test_explainer_integrated_gradient_1_sample(tokenizer, model, x, config):
    explainer = IntegratedGradientExplainer(tokenizer, model, riemann_samples=1, **config.args)

    im = (explainer(x, tf.constant([0, 1])) ** 2).to_tensor(default_value=-1).numpy()
    np.testing.assert_allclose(im, [
        [0, 4, 4, 0],
        [4, 0, 4, -1]
    ])

@pytest.mark.parametrize("config", compile_configs, ids=lambda config: config.name)
def test_explainer_integrated_gradient_2_samples(tokenizer, model, x, config):
    explainer = IntegratedGradientExplainer(tokenizer, model, riemann_samples=2, **config.args)

    im = (explainer(x, tf.constant([0, 1])) ** 2).to_tensor(default_value=-1).numpy()
    np.testing.assert_allclose(im, [
        [0, 1.5625, 1.5625, 0],
        [1.5625, 0, 1.5625, -1]
    ])
