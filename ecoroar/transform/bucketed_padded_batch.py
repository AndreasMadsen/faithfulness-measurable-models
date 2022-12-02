
import numpy as np
import tensorflow as tf

from typing import List, Callable
from ..types import TokenizedDict
from ..tokenizer._abstract_tokenizer import AbstractTokenizer

@tf.function
def _get_observation_length(x, y):
    return tf.shape(x['input_ids'], out_type=tf.dtypes.int32)[0]

class BucketedPaddedBatch:
    def __init__(self, datasets: List[tf.data.Dataset], quantiles: List[float] = [0.25, 0.5, 0.75, 0.9]):
        """Pads observations to fixed lengths using a quantile heuristic

        When using XLA JIT a new program needs to be compiled for every input shape.
        Using the maximum model input size or the maximum dataset size, often comes
        at a performance penalty. This heuristic creates len(quantiles) + 1 padding
        lengths, limiting the number of compiled programs while also keeping the
        input size small.

        Args:
            dataset: (tf.data.Dataset): the dataset to compute statistics on
            quantiles (List[float], optional): The quantiles to create split at
        """
        # concat datasets
        datasets_iter = iter(datasets)
        dataset_all = next(datasets_iter)
        for dataset_left in datasets_iter:
            dataset_all.concatenate(dataset_left)

        # get observation lengths
        lengths_dataset = dataset_all \
            .map(_get_observation_length, num_parallel_calls=tf.data.AUTOTUNE)
        lengths = np.fromiter(lengths_dataset.as_numpy_iterator(), dtype=np.int32)

        # bucket size via quantile heuristic
        self._bucket_boundaries = np.hstack((
            np.quantile(lengths, q=quantiles).astype(np.int32),
            np.amax(lengths)
        ))

    def __call__(self, batch_size, **kwargs) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
        """Pads the datasets to the infered bucket bounderies

        Args:
            batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
                consecutive elements of this dataset to combine in a single batch.
            padded_shapes: (Optional.) A (nested) structure of `tf.TensorShape` or
                `tf.int64` vector tensor-like objects representing the shape to which
                the respective component of each input element should be padded prior
                to batching. Any unknown dimensions will be padded to the maximum size
                of that dimension in each batch. If unset, all dimensions of all
                components are padded to the maximum size in the batch. `padded_shapes`
                must be set if any component has an unknown rank.
            padding_values: (Optional.) A (nested) structure of scalar-shaped
                `tf.Tensor`, representing the padding values to use for the respective
                components. None represents that the (nested) structure should be padded
                with default values.  Defaults are `0` for numeric types and the empty
                string for string types. The `padding_values` should have the same
                (nested) structure as the input dataset. If `padding_values` is a single
                element and the input dataset has multiple components, then the same
                `padding_values` will be used to pad every component of the dataset.
                If `padding_values` is a scalar, then its value will be broadcasted
                to match the shape of each component.

        Returns:
             Callable[[tf.data.Dataset], tf.data.Dataset]: function that maps from dataset to dataset
        """

        # NOTE: Might cause convertion issues to have mini batches of similar length,
        # consider doing something using .batch() followed by RaggedTensor.boundery_shape() and
        # RaggedTensor.to_tensor() instead.
        def padded_batch(dataset):
            return dataset.bucket_by_sequence_length(
                element_length_func=_get_observation_length,
                bucket_boundaries=self._bucket_boundaries,
                bucket_batch_sizes=[batch_size] * (len(self._bucket_boundaries) + 1),
                pad_to_bucket_boundary=True,
                **kwargs
            )

        return padded_batch
