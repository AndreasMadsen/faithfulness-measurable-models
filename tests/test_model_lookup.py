
import tensorflow as tf
import numpy as np
import pytest

from ecoroar.model import LookupTestModel
from ecoroar.tokenizer import SimpleTestTokenizer


@pytest.fixture
def tokenizer():
    return SimpleTestTokenizer()

@pytest.fixture
def dummy_input_tokenized():
    return {
        'input_ids': tf.constant([
            [0, 3, 3, 1, 2],
            [0, 3, 1, 2, 2]
        ], dtype=tf.dtypes.int32),
        'attention_mask': tf.constant([
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0]
        ], dtype=tf.dtypes.int8)
    }

def test_lookup_model_constructor(dummy_input_tokenized):
    values = tf.constant([[0.2, 0.2], [0.5, 0.4]], dtype=tf.dtypes.float32)

    model = LookupTestModel(keys=dummy_input_tokenized, values=values)
    model.compile()

    np.testing.assert_allclose(model(dummy_input_tokenized).logits.numpy(), values)
    np.testing.assert_allclose(model(dummy_input_tokenized).logits.numpy(), values)

def test_lookup_model_from_string(tokenizer, dummy_input_tokenized):
    model = LookupTestModel.from_string(
        tokenizer=tokenizer,
        mapping={
            '[BOS] token token [EOS]': (0.2, 0.2),
            '[BOS] token [EOS]': (0.5, 0.4)
        })
    model.compile()

    np.testing.assert_allclose(model(dummy_input_tokenized).logits.numpy(), [[0.2, 0.2], [0.5, 0.4]])
    np.testing.assert_allclose(model(dummy_input_tokenized).logits.numpy(), [[0.2, 0.2], [0.5, 0.4]])
