
from dataclasses import dataclass

import pytest
import tensorflow as tf
import numpy as np

from ecoroar.model import SimpleTestModel
from ecoroar.tokenizer import SimpleTestTokenizer
from ecoroar.explain import RandomExplainer


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

def test_explainer_random(tokenizer, model, x):
    explainer = RandomExplainer(tokenizer, model, seed=0)

    im = explainer(x, tf.constant([0, 1])).to_tensor(default_value=-1).numpy()
    np.testing.assert_allclose(im, [
        [0.311793, 0.826341, 0.684946, 0.006709],
        [0.390651, 0.292631, 0.992169, -1.     ]
    ], rtol=0.1)
