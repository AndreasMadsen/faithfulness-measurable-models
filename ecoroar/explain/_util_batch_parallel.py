
from typing import Union

import tensorflow as tf

def batch_parallel(batch_size: Union[int, tf.Tensor], num_parallel_calls=tf.data.AUTOTUNE):
    """This decorator will make any tf.function run on batches in parallel

    Unbatched arguments may be passed using `extra_args = tuple(arg_1, arg_1, ...)`.

    The resulting tensor(s) will be RaggedTensors when the output shape(s) from the
        decorated function can not infered to be the same.

    Note that the function will run on the CPU. Due to limitations with the
        underlying tf.data.Dataset executor.

    Args:
        batch_size (int): The maximum batch_size that will be encountered.
            A value to low will result in an runtime error.

    Example:
        @batch_parallel(16)
        @tf.function(reduce_retracing=True)
        def beam_select(beam_score, beam_size):
            return tf.argsort(beam_score, stable=True, direction='DESCENDING')[:beam_size]

        beam_select(tensor, extra_args=(4, ))
    """
    batch_size = tf.cast(batch_size, dtype=tf.dtypes.int64)

    def decorator(fn):
        @tf.function(reduce_retracing=True)
        def mapper(*batch_args, extra_args=tuple()):
            return tf.data.Dataset.from_tensor_slices(batch_args) \
                .map(lambda *args: fn(*args, *extra_args), num_parallel_calls=num_parallel_calls, deterministic=True) \
                .ragged_batch(batch_size) \
                .get_single_element()

        return mapper
    return decorator
