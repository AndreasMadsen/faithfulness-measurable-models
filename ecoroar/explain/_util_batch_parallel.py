
from typing import Union

import tensorflow as tf

def batch_parallel(batch_size: Union[int, tf.Tensor], num_parallel_calls=tf.data.AUTOTUNE, validate=True):
    """This decorator will make any tf.function run on batches in parallel

    Unbatched arguments may be passed using `extra_args = tuple(arg_1, arg_1, ...)`.

    The resulting tensor(s) will be RaggedTensors when the output shape(s) from the
        decorated function can not infered to be the same.

    Note that the function will run on the CPU. Due to limitations with the
        underlying tf.data.Dataset executor.

    Args:
        batch_size (int): The maximum batch_size that will be encountered.
            A value to low will result in an runtime error.
        num_parallel_calls (int, optional): the number of num_parallel_calls. By default AUTOTUNE.
        validate (bool, optional): Should the RaggedTensor normalization be validated. By default true.

    Example:
        @batch_parallel(16)
        @tf.function(reduce_retracing=True)
        def beam_select(beam_score, beam_size):
            return tf.argsort(beam_score, stable=True, direction='DESCENDING')[:beam_size]

        beam_select(tensor, extra_args=(4, ))
    """
    batch_size = tf.cast(batch_size, dtype=tf.dtypes.int64)
    def normalize(ragged):
        return tf.RaggedTensor.from_row_splits(
                values=tf.reshape(ragged.flat_values, tf.concat(([-1], ragged.bounding_shape()[2:]), axis=0)),
                row_splits=ragged.row_splits,
                validate=validate
            )

    def decorator(fn):
        @tf.function(reduce_retracing=True)
        def mapper(*batch_args, extra_args=tuple()):
            out = tf.data.Dataset.from_tensor_slices(batch_args) \
                .map(lambda *args: fn(*args, *extra_args), num_parallel_calls=num_parallel_calls, deterministic=True) \
                .ragged_batch(batch_size) \
                .get_single_element()

            out = tf.nest.map_structure(normalize, out)

            return out

        return mapper
    return decorator
