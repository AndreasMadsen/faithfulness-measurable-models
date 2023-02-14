
import pathlib

import pytest
import numpy as np
import tensorflow as tf

from ecoroar.tokenizer import HuggingfaceTokenizer
from ecoroar.transform import RandomMaxMasking


@pytest.fixture
def tokenizer():
    return HuggingfaceTokenizer('roberta-base', persistent_dir=pathlib.Path('.'))


@pytest.fixture
def single_obs_input(tokenizer):
    output = tokenizer(("This was an absolutely terrible movie.", ))
    return {
        'input_ids': tf.expand_dims(output['input_ids'], axis=0),
        'attention_mask': tf.expand_dims(output['attention_mask'], axis=0),
    }


@pytest.fixture
def simple_dataset(tokenizer):
    return tf.data.Dataset.from_tensor_slices([
        "This was truely an absolutely - terrible movie.",
        "This was an terrible movie."
    ]).map(lambda doc: tokenizer((doc, )))


def test_masking_special_tokens_kept(tokenizer, single_obs_input):
    for seed in range(100):
        masker_max = RandomMaxMasking(1, tokenizer, seed=seed)

        np.testing.assert_array_equal(single_obs_input['input_ids'].numpy(),
                                      [[0, 713, 21, 41, 3668, 6587, 1569, 4, 2]])

        # masker_max samples a random masking ratio between 0 and 1.
        # Therefore not all tokens will be masked.
        masked_output = masker_max(single_obs_input)['input_ids']
        assert masked_output.numpy()[0, 0] == tokenizer.bos_token_id.numpy()
        assert masked_output.numpy()[0, -1] == tokenizer.eos_token_id.numpy()


def test_masking_zero(tokenizer, single_obs_input):
    for seed in range(100):
        masker_none = RandomMaxMasking(0, tokenizer, seed=seed)

        np.testing.assert_array_equal(single_obs_input['input_ids'].numpy(),
                                      [[0, 713, 21, 41, 3668, 6587, 1569, 4, 2]])

        np.testing.assert_array_equal(masker_none(single_obs_input)['input_ids'].numpy(),
                                      [[0, 713, 21, 41, 3668, 6587, 1569, 4, 2]])


def test_masking_some(tokenizer, single_obs_input):
    masker_some = RandomMaxMasking(0.5, tokenizer, seed=1)
    mask = tokenizer.mask_token_id.numpy()

    np.testing.assert_array_equal(single_obs_input['input_ids'].numpy(),
                                  [[0, 713, 21, 41, 3668, 6587, 1569, 4, 2]])

    np.testing.assert_array_equal(masker_some(single_obs_input)['input_ids'].numpy(),
                                  [[0, mask, 21, 41, mask, 6587, 1569, mask, 2]])


def test_batch_masking(tokenizer):
    masker = RandomMaxMasking(0.8, tokenizer, seed=2)
    mask = tokenizer.mask_token_id.numpy()
    pad = tokenizer.pad_token_id.numpy()

    inputs = {
        'input_ids': tf.constant([[0, 713, 21, 41, 3668, 6587, 1569,   4,   2, pad],
                                  [0, 713, 21, 41,    4,    2,  pad, pad, pad, pad]], dtype=tf.int32),
        'attention_mask': tf.constant([[1,   1,  1,  1,    1,    1,    1,   1,   1, 0],
                                       [1,   1,  1,  1,    1,    1,    0,   0,   0, 0]], dtype=tf.int8)
    }
    outputs = masker(inputs)

    np.testing.assert_array_equal(outputs['input_ids'].numpy(),
                                  [[0, mask, mask, mask, mask, 6587, mask,   4,   2, pad],
                                   [0,  713,   21,   41, mask,    2,  pad, pad, pad, pad]])

    np.testing.assert_array_equal(outputs['attention_mask'].numpy(),
                                  inputs['attention_mask'].numpy())


def test_masking_partially_known_shape(tokenizer, simple_dataset):
    mask = tokenizer.mask_token_id.numpy()

    masker_40 = RandomMaxMasking(0.6, tokenizer, seed=1)
    masked_1, masked_2 = simple_dataset \
        .padded_batch(1, padding_values=tokenizer.padding_values) \
        .map(lambda x: masker_40(x)['input_ids']).as_numpy_iterator()
    np.testing.assert_array_equal(masked_1, [[0, mask, 21, 1528, mask,   41, mask, mask, 6587, mask, 4, 2]])
    np.testing.assert_array_equal(masked_2, [[0,  713, 21,   41, 6587, mask, mask, 2]])


def test_masking_batch_dataset(tokenizer, simple_dataset):
    mask = tokenizer.mask_token_id.numpy()

    masker_40 = RandomMaxMasking(0.8, tokenizer, seed=2)
    masked, = simple_dataset \
        .padded_batch(2, padding_values=tokenizer.padding_values) \
        .map(lambda x: masker_40(x)['input_ids']).as_numpy_iterator()
    np.testing.assert_array_equal(masked, [[0, mask, mask, mask, mask,   41, mask, 111, mask, 1569, mask, 2],
                                           [0,  713, mask,   41, 6587, 1569,    4,   2,    1,    1,    1, 1]])
