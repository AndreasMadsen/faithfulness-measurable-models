
from typing import Callable

import tensorflow as tf

class MapOnGPU:
    def __init__(self, mapper) -> None:
        self._mapper = mapper

    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Map the dataset using the provided mapper function on the GPU.

        Args:
            dataset (tf.data.Dataset): the dataset to be mapped.

        Returns:
           tf.data.Dataset: mapped dataset
        """
        # tf.Dataset.from_generator is used because tf.Dataset.map will run on the CPU,
        # and the explainer is likely to do model inference and perhaps derivatives.
        # Those are expensive calculations, and should be done on the GPU. using
        # `with tf.device('GPU')` to force calculations on the GPU is an option too.
        # However, this is still 2x-3x slower, likely because data is not prefetched to
        # the GPU.
        def _generator():
            for args in dataset.prefetch(tf.data.AUTOTUNE):
                yield self._mapper(*args)

        return tf.data.Dataset.from_generator(
            _generator,
            output_signature=tf.data.experimental.get_structure(dataset)
        ).apply(tf.data.experimental.assert_cardinality(dataset.cardinality()))
