
import pathlib

import pytest
import numpy as np
import tensorflow as tf

from ecoroar.tokenizer import HuggingfaceTokenizer
from ecoroar.transform import TransformSampler, RandomMaxMasking, RandomFixedMasking


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
        "This was an terrible movie.",
        "This was an terrible movie.",
        "This was truely an absolutely - terrible movie.",
    ]).map(lambda doc: tokenizer((doc, )))


def test_transform_sampler_deterministic(tokenizer, simple_dataset):
    mask = tokenizer.mask_token_id.numpy()
    masker = TransformSampler([
        RandomMaxMasking(0.0, tokenizer, seed=0),
        RandomFixedMasking(1.0, tokenizer, seed=0)
    ], seed=0)

    masked_1, masked_2 = simple_dataset \
        .padded_batch(2, padding_values=tokenizer.padding_values) \
        .map(lambda x: masker(x)['input_ids']).as_numpy_iterator()

    np.testing.assert_array_equal(masked_1, [[0,  713,   21, 1528,  352,   41, 3668, 111, 6587, 1569, 4, 2],
                                             [0, mask, mask, mask, mask, mask, mask,   2,    1,    1, 1, 1]])

    np.testing.assert_array_equal(masked_2, [[0,  713,   21,   41, 6587, 1569,    4,    2,    1,    1,    1, 1],
                                             [0, mask, mask, mask, mask, mask, mask, mask, mask, mask, mask, 2]])

def test_transform_sampler_stocastic(tokenizer, single_obs_input):
    mask = tokenizer.mask_token_id.numpy()
    source_sequence = np.asarray([[0, 713, 21, 41, 3668, 6587, 1569, 4, 2]])
    masked_sequence = np.asarray([[0, mask, mask, mask, mask, mask, mask, mask, 2]])

    np.testing.assert_array_equal(single_obs_input['input_ids'].numpy(), source_sequence)
    num_of_masked = 0
    num_of_unmasked = 0

    for seed in range(100):
        masker = TransformSampler([
            RandomMaxMasking(0.0, tokenizer, seed=seed),
            RandomFixedMasking(1.0, tokenizer, seed=seed)
        ], seed=seed, stochastic=True)
        maybe_masked = masker(single_obs_input)['input_ids'].numpy()

        if np.array_equal(maybe_masked, source_sequence):
            num_of_masked += 1
        elif np.array_equal(maybe_masked, masked_sequence):
            num_of_unmasked += 1
        else:
            raise ValueError(f'unexpected output: {maybe_masked}')

    assert num_of_masked + num_of_unmasked == 100
    assert num_of_masked == 47  # seed specific
    assert num_of_unmasked == 53  # seed specific
