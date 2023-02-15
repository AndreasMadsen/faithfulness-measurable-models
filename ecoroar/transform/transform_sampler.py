
from typing import List

import tensorflow as tf

from ..types import TokenizedDict, InputTransform


def _slice_structure(structure, slicer):
    return tf.nest.map_structure(lambda tensor: tensor[slicer], structure)

def _concat_structure(structures, **kwargs):
    return tf.nest.map_structure(lambda *tensors: tf.concat(tensors, **kwargs), *structures)

class TransformSampler:
    def __init__(self, transforms: List[InputTransform], stochastic=False, seed: int = None):
        """_summary_

        Args:
            transforms (List[InputTransform]): A list of transforms to sample from.
            stochastic (bool, optional): If true, this will sample which transform to use. Defaults to False.
            seed (int, optional): Seed used to select which transform to use. Defaults to None.
        """
        if len(transforms) != 2:
            raise ValueError('currently only two transforms is supported')

        self._transforms = transforms
        self._stochastic = stochastic

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
        input_ids = tf.ensure_shape(x['input_ids'], [None, None])
        batch_size, _ = tf.unstack(tf.shape(input_ids), num=2)

        if self._stochastic:
            selector = tf.cast(
                self._rng.uniform((batch_size, 1), minval=0, maxval=2, dtype=tf.dtypes.int32),
                dtype=tf.dtypes.bool)
            return tf.nest.map_structure(
                lambda t0, t1: tf.where(selector, t0, t1),
                self._transforms[0](x),
                self._transforms[1](x)
            )
        else:
            return _concat_structure([
                self._transforms[0](_slice_structure(x, (slice(None, batch_size // 2), ...))),
                self._transforms[1](_slice_structure(x, (slice(batch_size // 2, None), ...)))
            ], axis=0)
