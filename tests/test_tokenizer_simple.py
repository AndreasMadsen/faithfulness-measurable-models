
import pytest
import tensorflow as tf
import numpy as np

from ecoroar.tokenizer import SimpleTestTokenizer


@pytest.fixture
def text_dataset():
    return tf.data.Dataset.from_tensor_slices([
        '[BOS] token token [EOS] [PAD]',
        '[BOS] token [EOS] [PAD] [PAD]',
    ])


def test_simple_tokenizer(text_dataset):
    tokenizer = SimpleTestTokenizer()

    inputs = text_dataset \
        .map(lambda doc: tokenizer((doc, ))) \
        .batch(2) \
        .get_single_element()

    np.testing.assert_array_equal(inputs['attention_mask'], [
        [1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0]
    ])
    np.testing.assert_array_equal(inputs['input_ids'], [
        [0, 3, 3, 1, 2],
        [0, 3, 1, 2, 2]
    ])
