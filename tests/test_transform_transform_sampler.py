
import pathlib

import numpy as np

from ecoroar.tokenizer import HuggingfaceTokenizer
from ecoroar.transform import TransformSampler, RandomMaxMasking, RandomFixedMasking


def test_transform_sampler():
    tokenizer = HuggingfaceTokenizer('roberta-base', persistent_dir=pathlib.Path('.'))
    output = tokenizer(("This was an absolutely terrible movie.", ))

    mask = 50264
    source_sequence = np.asarray([0, 713, 21, 41, 3668, 6587, 1569, 4, 2])
    masked_sequence = np.asarray([0, mask, mask, mask, mask, mask, mask, mask, 2])

    np.testing.assert_array_equal(output['input_ids'].numpy(), source_sequence)
    num_of_masked = 0
    num_of_unmasked = 0

    for seed in range(100):
        masker = TransformSampler([
            RandomMaxMasking(0.0, tokenizer, seed=seed),
            RandomFixedMasking(1.0, tokenizer, seed=seed)
        ], seed=seed)
        maybe_masked = masker(output)['input_ids'].numpy()

        if np.array_equal(maybe_masked, source_sequence):
            num_of_masked += 1
        elif np.array_equal(maybe_masked, masked_sequence):
            num_of_unmasked += 1
        else:
            raise ValueError(f'unexpected output: {maybe_masked}')

    assert num_of_masked + num_of_unmasked == 100
    assert num_of_masked == 53  # seed specific
    assert num_of_unmasked == 47  # seed specific
