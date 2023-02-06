
import pathlib

import numpy as np

from ecoroar.tokenizer import HuggingfaceTokenizer
from ecoroar.transform import RandomMaxMasking


def test_masking_special_tokens_kept():
    tokenizer = HuggingfaceTokenizer('roberta-base', persistent_dir=pathlib.Path('.'))
    output = tokenizer(("This was an absolutely terrible movie.", ))

    for seed in range(100):
        masker_max = RandomMaxMasking(1.0, tokenizer, seed=0)

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
        masker_none = RandomMaxMasking(0.0, tokenizer, seed=0)

        np.testing.assert_array_equal(output['input_ids'].numpy(),
                                      [0, 713, 21, 41, 3668, 6587, 1569, 4, 2])

        np.testing.assert_array_equal(masker_none(output)['input_ids'].numpy(),
                                      [0, 713, 21, 41, 3668, 6587, 1569, 4, 2])


def test_masking_some():
    tokenizer = HuggingfaceTokenizer('roberta-base', persistent_dir=pathlib.Path('.'))
    output = tokenizer(("This was an absolutely terrible movie.", ))
    masker_some = RandomMaxMasking(0.5, tokenizer, seed=1)
    mask = tokenizer.mask_token_id.numpy()

    np.testing.assert_array_equal(output['input_ids'].numpy(),
                                  [0, 713, 21, 41, 3668, 6587, 1569, 4, 2])

    np.testing.assert_array_equal(masker_some(output)['input_ids'].numpy(),
                                  [0, mask, 21, 41, mask, 6587, 1569, mask, 2])
