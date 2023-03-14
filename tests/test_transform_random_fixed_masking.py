
import pathlib

import pytest
import numpy as np
import tensorflow as tf

from ecoroar.tokenizer import HuggingfaceTokenizer
from ecoroar.transform import RandomFixedMasking


@pytest.fixture
def tokenizer():
    return HuggingfaceTokenizer('roberta-base', persistent_dir=pathlib.Path('.'))


@pytest.fixture
def single_obs_input(tokenizer):
    output = tokenizer(("This was truely an absolutely - terrible movie.", ))
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
    mask = tokenizer.mask_token_id.numpy()
    for seed in range(10):
        masker_max = RandomFixedMasking(1, tokenizer, seed=seed)

        np.testing.assert_array_equal(single_obs_input['input_ids'].numpy(),
                                      [[0, 713, 21, 1528, 352, 41, 3668, 111, 6587, 1569, 4, 2]])

        # masker_max samples a random masking ratio between 0 and 1.
        # Therefore not all tokens will be masked.
        np.testing.assert_array_equal(masker_max(single_obs_input)['input_ids'].numpy(),
                                     [[0, mask, mask, mask, mask, mask, mask, mask, mask, mask, mask, 2]])
        masked_output = masker_max(single_obs_input)['input_ids']
        assert masked_output.numpy()[0, 0] == tokenizer.bos_token_id.numpy()
        assert masked_output.numpy()[0, -1] == tokenizer.eos_token_id.numpy()


def test_masking_zero(tokenizer, single_obs_input):
    for seed in range(10):
        masker_none = RandomFixedMasking(0, tokenizer, seed=seed)

        np.testing.assert_array_equal(single_obs_input['input_ids'].numpy(),
                                      [[0, 713, 21, 1528, 352, 41, 3668, 111, 6587, 1569, 4, 2]])

        np.testing.assert_array_equal(masker_none(single_obs_input)['input_ids'].numpy(),
                                      [[0, 713, 21, 1528, 352, 41, 3668, 111, 6587, 1569, 4, 2]])


def test_masking_some(tokenizer, single_obs_input):
    mask = tokenizer.mask_token_id.numpy()

    np.testing.assert_array_equal(single_obs_input['input_ids'].numpy(),
                                  [[0, 713, 21, 1528, 352, 41, 3668, 111, 6587, 1569, 4, 2]])

    masker_20 = RandomFixedMasking(0.2, tokenizer, seed=1)
    np.testing.assert_array_equal(masker_20(single_obs_input)['input_ids'].numpy(),
                                  [[0, 713, 21, 1528, 352, 41, 3668, mask, mask, 1569, 4, 2]])

    masker_40 = RandomFixedMasking(0.4, tokenizer, seed=1)
    np.testing.assert_array_equal(masker_40(single_obs_input)['input_ids'].numpy(),
                                  [[0, 713, 21, 1528, 352, mask, mask, mask, mask, 1569, 4, 2]])

    masker_60 = RandomFixedMasking(0.6, tokenizer, seed=1)
    np.testing.assert_array_equal(masker_60(single_obs_input)['input_ids'].numpy(),
                                  [[0, mask, 21, 1528, 352, mask, mask, mask, mask, 1569, mask, 2]])

    masker_80 = RandomFixedMasking(0.8, tokenizer, seed=1)
    np.testing.assert_array_equal(masker_80(single_obs_input)['input_ids'].numpy(),
                                  [[0, mask, 21, mask, mask, mask, mask, mask, mask, 1569, mask, 2]])


def test_masking_partially_known_shape(tokenizer, simple_dataset):
    mask = tokenizer.mask_token_id.numpy()

    masker_40 = RandomFixedMasking(0.4, tokenizer, seed=1)
    masked_1, masked_2 = simple_dataset \
        .padded_batch(1, padding_values=tokenizer.padding_values) \
        .map(lambda x: masker_40(x)['input_ids']).as_numpy_iterator()
    np.testing.assert_array_equal(masked_1, [[0, 713, 21, 1528, 352, mask, mask, mask, mask, 1569, 4, 2]])
    np.testing.assert_array_equal(masked_2, [[0, 713, 21, mask, mask, 1569, 4, 2]])


def test_masking_batch_dataset(tokenizer, simple_dataset):
    mask = tokenizer.mask_token_id.numpy()

    masker_40 = RandomFixedMasking(0.4, tokenizer, seed=1)
    masked, = simple_dataset \
        .padded_batch(2, padding_values=tokenizer.padding_values) \
        .map(lambda x: masker_40(x)['input_ids']).as_numpy_iterator()
    np.testing.assert_array_equal(masked, [[0, 713, 21, 1528, 352, mask, mask, mask, mask, 1569, 4, 2],
                                           [0, 713, 21, mask, mask, 1569,   4,    2,    1,    1, 1, 1]])
