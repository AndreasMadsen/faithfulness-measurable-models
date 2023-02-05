
import numpy as np
import tensorflow as tf

from typing import List, Callable
from ..types import TokenizedDict, InputTransform
from ..tokenizer._abstract_tokenizer import AbstractTokenizer

@tf.function
def _get_bounding_shape(x, y):
    return x['input_ids'].bounding_shape(out_type=tf.dtypes.int32)

class BucketedPaddedBatch(InputTransform):
    def __init__(self, datasets: List[tf.data.Dataset],
                 quantiles: List[float] = [0.25, 0.5, 0.75, 0.9],
                 batch_size: int=16,
                 bounding_shape: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]=_get_bounding_shape):
        """Pads observations to fixed lengths using a quantile heuristic

        When using XLA JIT a new program needs to be compiled for every input shape.
        Using the maximum model input size or the maximum dataset size, often comes
        at a performance penalty. This heuristic creates len(quantiles) + 1 padding
        lengths, limiting the number of compiled programs while also keeping the
        input size small.

        Args:
            datasets: (List[tf.data.Dataset]): the dataset to compute statistics on
            quantiles (List[float], optional): The quantiles to create split at
            batch_size (int, optional): The batch_size to compute batched sequence-length from
            bounding_shape (Callable[[tf.Tensor, tf.Tensor], tf.Tensor], optional): Function to
                get the bounding shape of the input. Shound return [batch_size, batch_sequence_length] as
                tf.Tensor.
                Defaults to using lambda x, y: x['input_ids'].bounding_shape(out_type=tf.dtypes.int32)
        """
        self._bounding_shape = bounding_shape

        lengths_list = []
        for dataset in datasets:
            # get observation lengths
            # tf.data.experimental.dense_to_ragged_batch becomes
            # tf.data.Dataset.ragged_batch(num_parallel_calls=tf.data.AUTOTUNE) in TF v2.11.0
            lengths_dataset = dataset \
                .apply(tf.data.experimental.dense_to_ragged_batch(batch_size)) \
                .map(lambda *args: bounding_shape(*args)[-1], num_parallel_calls=tf.data.AUTOTUNE)
            lengths_list.append(
                np.fromiter(lengths_dataset.as_numpy_iterator(), dtype=np.int32)
            )
        lengths = np.hstack(lengths_list)

        # bucket size via quantile heuristic
        self._bucket_boundaries = tf.convert_to_tensor(np.unique(np.hstack((
            np.quantile(lengths, q=quantiles, method='closest_observation'),
            np.amax(lengths)
        ))))

    @property
    def bounderies(self):
        return self._bucket_boundaries

    @tf.function
    def _pad_tensor(self, ragged_tensor, padding_value, target_shape):
        # likely tf.Tensor
        if padding_value is None:
            return ragged_tensor

        # likely tf.RaggedTensor
        return ragged_tensor.to_tensor(default_value=padding_value, shape=target_shape)

    @tf.function
    def _pad_structure(self, inputs, padding_values):
        batch_size, batch_sequence_length = tf.unstack(self._bounding_shape(*inputs), num=2)

        # Round up the batch_sequence_length to the bucket_boundary
        batch_sequence_length_upper = self._bucket_boundaries[
            tf.searchsorted(self._bucket_boundaries, [batch_sequence_length])[0]
        ]
        target_shape = tf.stack([batch_size, batch_sequence_length_upper])

        # Pad all tensors that needs padding.
        return tf.nest.map_structure(
            lambda tensor, padding_value: self._pad_tensor(tensor, padding_value, target_shape),
            inputs, padding_values
        )

    def __call__(self, batch_size, padding_values=None,
                 deterministic=None, num_parallel_calls=None,
                 **kwargs) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
        """Pads the datasets to the infered bucket bounderies

        Args:
            batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
                consecutive elements of this dataset to combine in a single batch.
            padding_values: A (nested) structure of scalar-shaped
                `tf.Tensor`, representing the padding values to use for the respective
                components. None represents that the (nested) structure should be padded
                with default values.  Defaults are `0` for numeric types and the empty
                string for string types. The `padding_values` should have the same
                (nested) structure as the input dataset. If `padding_values` is a single
                element and the input dataset has multiple components, then the same
                `padding_values` will be used to pad every component of the dataset.
                If `padding_values` is a scalar, then its value will be broadcasted
                to match the shape of each component.
            padded_shapes: (Optional.) A (nested) structure of `tf.TensorShape` or
                `tf.int64` vector tensor-like objects representing the shape to which
                the respective component of each input element should be padded prior
                to batching. Any unknown dimensions will be padded to the maximum size
                of that dimension in each batch. If unset, all dimensions of all
                components are padded to the maximum size in the batch. `padded_shapes`
                must be set if any component has an unknown rank.
            num_parallel_calls: (Optional.) A `tf.int64` scalar `tf.Tensor`,
                representing the number of batches to compute asynchronously in
                parallel.
                If not specified, batches will be computed sequentially. If the value
                `tf.data.AUTOTUNE` is used, then the number of parallel
                calls is set dynamically based on available resources.
            deterministic: (Optional.) When `num_parallel_calls` is specified, if this
                boolean is specified (`True` or `False`), it controls the order in which
                the transformation produces elements. If set to `False`, the
                transformation is allowed to yield elements out of order to trade
                determinism for performance. If not specified, the
                `tf.data.Options.deterministic` option (`True` by default) controls the
                behavior.

        Returns:
             Callable[[tf.data.Dataset], tf.data.Dataset]: function that maps from dataset to dataset
        """

        if padding_values is None:
            raise ValueError('padding_values must be specified')

        # tf.data.experimental.dense_to_ragged_batch becomes
        # tf.data.Dataset.ragged_batch(deterministic=deterministic, num_parallel_calls=num_parallel_calls)
        # in TF v2.11.0
        def padded_batch(dataset):
            return dataset \
                .apply(tf.data.experimental.dense_to_ragged_batch(batch_size)) \
                .map(lambda *args: self._pad_structure(args, padding_values),
                     deterministic=deterministic,
                     num_parallel_calls=num_parallel_calls)

        return padded_batch
