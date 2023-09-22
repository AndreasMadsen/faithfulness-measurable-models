
from typing import Callable, Any

import tensorflow as tf


class MapOnGPU:
    def __init__(self, mapper: Callable[..., Any], output_signature: Callable[[tf.data.Dataset], Any]) -> None:
        self._mapper = mapper
        self._output_signature = output_signature

    def __call__(self, dataset_in: tf.data.Dataset) -> tf.data.Dataset:
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
        if isinstance(tf.data.experimental.get_structure(dataset_in), tuple):
            def _generator():
                for args in dataset_in.prefetch(tf.data.AUTOTUNE):
                    yield self._mapper(*args)
        else:
            def _generator():
                for arg in dataset_in.prefetch(tf.data.AUTOTUNE):
                    yield self._mapper(arg)

        dataset_out = tf.data.Dataset.from_generator(
            _generator,
            output_signature=self._output_signature(dataset_in)
        )
        cardinality = dataset_in.cardinality()
        if cardinality >= 0:
            dataset_out = dataset_out.apply(tf.data.experimental.assert_cardinality(cardinality))
        return dataset_out
