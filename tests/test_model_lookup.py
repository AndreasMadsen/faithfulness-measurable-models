
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


@pytest.fixture
def dummy_input_mapping():
    return {
        '[BOS] token  [EOS]  [PAD]': (-2.0, 2.0),
        '[BOS] [MASK] [EOS]  [PAD]': (-3.0, 3.0),

        '[BOS] token  token  [EOS]': (2.0, -2.0),
        '[BOS] token  [MASK] [EOS]': (1.0, -1.0),
        '[BOS] [MASK] token  [EOS]': (-1.0, 1.0),
        '[BOS] [MASK] [MASK] [EOS]': (1.0, -1.0),

        # B-1: 3      1      4      2
        # B-2: 4      3      2      1
        '[BOS] token  token  token  token  [EOS]': (5.0, 0.0),
        '[BOS] [MASK] [MASK] [MASK] [MASK] [EOS]': (2.0, 0.0),

        #      [MASK]
        '[BOS] [MASK] token  token  token  [EOS]': (2.0, 0.0),  # 1:3
        #      [MASK] [MASK]
        '[BOS] [MASK] [MASK] token  token  [EOS]': (3.0, 0.0),  # 12:5
        '[BOS] [MASK] [MASK] [MASK] token  [EOS]': (3.0, 0.0),  # 123:7, 312:5
        '[BOS] [MASK] [MASK] token  [MASK] [EOS]': (5.0, 0.0),  # 124:5
        #      [MASK]        [MASK]
        '[BOS] [MASK] token  [MASK] token  [EOS]': (6.0, 0.0),  # 31:3, 13:2
        '[BOS] [MASK] token  [MASK] [MASK] [EOS]': (2.0, 0.0),  # 314:6
        #      [MASK]               [MASK]
        '[BOS] [MASK] token  token  [MASK] [EOS]': (7.0, 0.0),  # 14:1

        #             [MASK]
        '[BOS] token  [MASK] token  token  [EOS]': (9.0, 0.0),  # 2:-4
        #             [MASK] [MASK]
        '[BOS] token  [MASK] [MASK] token  [EOS]': (8.0, 0.0),  # 3:4, 32:1
        '[BOS] token  [MASK] [MASK] [MASK] [EOS]': (0.0, 0.0),  # lowest
        #             [MASK]        [MASK]
        '[BOS] token  [MASK] token  [MASK] [EOS]': (2.0, 0.0),

        #                    [MASK]
        '[BOS] token  token  [MASK] token  [EOS]': (1.0, 0.0),  # 3:4
        #                    [MASK] [MASK]
        '[BOS] token  token  [MASK] [MASK] [EOS]': (8.0, 0.0),  # 34:1

        #                           [MASK]
        '[BOS] token  token  token  [MASK] [EOS]': (5.0, 0.0),  # 4:0
    }


def test_lookup_model_constructor(dummy_input_tokenized):
    values = tf.constant([[0.2, 0.2], [0.5, 0.4]], dtype=tf.dtypes.float32)

    model = LookupTestModel(keys=dummy_input_tokenized, values=values)
    model.compile()

    np.testing.assert_allclose(model(dummy_input_tokenized).logits.numpy(), values)
    np.testing.assert_allclose(model(dummy_input_tokenized).logits.numpy(), values)


def test_lookup_model_from_string(tokenizer, dummy_input_mapping):
    mapping_tf = [
        (key, tf.convert_to_tensor(value, dtype=tf.dtypes.float32))
        for key, value in dummy_input_mapping.items()
    ]

    model = LookupTestModel.from_string(tokenizer=tokenizer, mapping=dummy_input_mapping)
    model.compile()

    keys, values = tf.data.experimental.from_list(mapping_tf) \
        .map(lambda doc, logit: (tokenizer((doc, )), logit)) \
        .padded_batch(len(dummy_input_mapping), padding_values=(tokenizer.padding_values, None)) \
        .get_single_element()

    np.testing.assert_allclose(model(keys).logits.numpy(), values)
    np.testing.assert_allclose(model(keys).logits.numpy(), values)
