
import pathlib

import numpy as np
import tensorflow as tf

from ecoroar.tokenizer import HuggingfaceTokenizer
from ecoroar.transform import RandomFixedMasking


def test_masking_special_tokens_kept():
    tokenizer = HuggingfaceTokenizer('roberta-base', persistent_dir=pathlib.Path('.'))
    output = tokenizer(("This was an absolutely terrible movie.", ))

    for seed in range(100):
        masker_max = RandomFixedMasking(1, tokenizer, seed=seed)

        np.testing.assert_array_equal(output['input_ids'].numpy(),
                                      [0, 713, 21, 41, 3668, 6587, 1569, 4, 2])

        # masker_max samples a random masking ratio between 0 and 1.
        # Therefore not all tokens will be masked.
        masked_output = masker_max(output)['input_ids']
        assert masked_output.numpy()[0] == tokenizer.bos_token_id.numpy()
        assert masked_output.numpy()[-1] == tokenizer.eos_token_id.numpy()


def test_masking_zero():
    tokenizer = HuggingfaceTokenizer('roberta-base', persistent_dir=pathlib.Path('.'))
    output = tokenizer(("This was an absolutely terrible movie.", ))

    for seed in range(100):
        masker_none = RandomFixedMasking(0, tokenizer, seed=seed)

        np.testing.assert_array_equal(output['input_ids'].numpy(),
                                      [0, 713, 21, 41, 3668, 6587, 1569, 4, 2])

        np.testing.assert_array_equal(masker_none(output)['input_ids'].numpy(),
                                      [0, 713, 21, 41, 3668, 6587, 1569, 4, 2])


def test_masking_some():
    tokenizer = HuggingfaceTokenizer('roberta-base', persistent_dir=pathlib.Path('.'))
    output = tokenizer(("This was truely an absolutely - terrible movie.", ))
    mask = tokenizer.mask_token_id.numpy()

    np.testing.assert_array_equal(output['input_ids'].numpy(),
                                  [0, 713, 21, 1528, 352, 41, 3668, 111, 6587, 1569, 4, 2])

    masker_20 = RandomFixedMasking(0.2, tokenizer, seed=1)
    np.testing.assert_array_equal(masker_20(output)['input_ids'].numpy(),
                                  [0, 713, 21, 1528, 352, 41, 3668, mask, mask, 1569, 4, 2])

    masker_40 = RandomFixedMasking(0.4, tokenizer, seed=1)
    np.testing.assert_array_equal(masker_40(output)['input_ids'].numpy(),
                                  [0, 713, 21, 1528, 352, mask, mask, mask, mask, 1569, 4, 2])

    masker_60 = RandomFixedMasking(0.6, tokenizer, seed=1)
    np.testing.assert_array_equal(masker_60(output)['input_ids'].numpy(),
                                  [0, mask, 21, 1528, 352, mask, mask, mask, mask, 1569, mask, 2])

    masker_80 = RandomFixedMasking(0.8, tokenizer, seed=1)
    np.testing.assert_array_equal(masker_80(output)['input_ids'].numpy(),
                                  [0, mask, 21, mask, mask, mask, mask, mask, mask, 1569, mask, 2])


def test_masking_partially_known_shape():
    tokenizer = HuggingfaceTokenizer('roberta-base', persistent_dir=pathlib.Path('.'))
    mask = tokenizer.mask_token_id.numpy()

    dataset = tf.data.Dataset.from_tensor_slices([
        "This was truely an absolutely - terrible movie.",
        "This was an terrible movie."
    ]).map(lambda doc: tokenizer((doc, )))

    masker_40 = RandomFixedMasking(0.4, tokenizer, seed=1)
    masked_1, masked_2 = dataset.map(lambda x: masker_40(x)['input_ids']).as_numpy_iterator()
    np.testing.assert_array_equal(masked_1, [0, 713, 21, 1528, 352, mask, mask, mask, mask, 1569, 4, 2])
    np.testing.assert_array_equal(masked_2, [0, 713, 21, mask, mask, 1569, 4, 2])
