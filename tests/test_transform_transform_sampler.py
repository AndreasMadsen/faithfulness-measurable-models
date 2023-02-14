
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
def simple_dataset(tokenizer):
    return tf.data.Dataset.from_tensor_slices([
        "This was truely an absolutely - terrible movie.",
        "This was an terrible movie.",
        "This was an terrible movie.",
        "This was truely an absolutely - terrible movie.",
    ]).map(lambda doc: tokenizer((doc, )))


def test_transform_sampler(tokenizer, simple_dataset):
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
