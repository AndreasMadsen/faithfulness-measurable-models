
from typing import List

import tensorflow as tf

from ..types import TokenizedDict, InputTransform


class TransformSampler:
    def __init__(self, transforms: List[InputTransform], seed: int = None):
        """_summary_

        Args:
            transforms (List[InputTransform]): A list of transforms to sample from.
            seed (int, optional): Seed used to select which transform to use. Defaults to None.
        """
        if len(transforms) != 2:
            raise ValueError('currently only two transforms is supported')

        self._transforms = transforms

        if seed is None:
            self._rng = tf.random.Generator.from_non_deterministic_state()
        else:
            self._rng = tf.random.Generator.from_seed(seed)

    @tf.function
    def __call__(self, x: TokenizedDict) -> TokenizedDict:
        """Randomly sample a transformer and use it.

        Args:
            x (TokenizedDict): Tokenized input

        Returns:
            TokenizedDict: transformed input
        """

        # BUG: test output is always the same
        random_switch = self._rng.uniform([1], minval=0, maxval=2, dtype=tf.dtypes.int32)[0]
        if random_switch == 0:
            return self._transforms[0](x)
        else:
            return self._transforms[1](x)
