
from typing import Union

import tensorflow as tf

def _get_batch_size(batch_args, out_type=tf.dtypes.int32):
    while isinstance(batch_args, tuple):
        batch_args = batch_args[0]
    return tf.shape(batch_args, out_type=out_type)[0]

def batch_parallel(batch_size: Union[int, tf.Tensor], num_parallel_calls=tf.data.AUTOTUNE, validate=True):
    """This decorator will make any tf.function run on batches in parallel

    Unbatched arguments may be passed using `extra_args = tuple(arg_1, arg_1, ...)`.

    The resulting tensor(s) will be RaggedTensors when the output shape(s) from the
        decorated function can not infered to be the same.

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
    if num_parallel_calls == tf.data.AUTOTUNE:
        num_parallel_calls = batch_size

    def decorator(fn):

        @tf.function(reduce_retracing=True)
        def mapper(*batch_args, extra_args=tuple()):
            batch_size = _get_batch_size(batch_args, out_type=tf.dtypes.int32)

            @tf.function(reduce_retracing=True)
            def wrap_fn(idx):
                return fn(*tf.nest.map_structure(lambda item: item[idx], batch_args), *extra_args)

            # prepear TensorArrays using the concrete_function
            structured_outputs = wrap_fn.get_concrete_function(tf.TensorSpec([], tf.dtypes.int32)).structured_outputs
            ta_data = tf.nest.map_structure(
                lambda item: tf.TensorArray(item.dtype, size=batch_size, infer_shape=False, element_shape=item.shape),
                structured_outputs)
            ta_length = tf.nest.map_structure(
                lambda item: tf.TensorArray(tf.dtypes.int64, size=batch_size, infer_shape=False, element_shape=[]),
                structured_outputs)

            # Run each observation in the batch
            for obs_i in tf.range(batch_size):
                tf.autograph.experimental.set_loop_options(
                    parallel_iterations=num_parallel_calls,
                    maximum_iterations=batch_size
                )

                out = wrap_fn(obs_i)
                ta_data = tf.nest.map_structure(
                    lambda item, ta: ta.write(obs_i, item),
                    out, ta_data)
                ta_length = tf.nest.map_structure(
                    lambda item, ta: ta.write(obs_i, tf.shape(item, out_type=tf.dtypes.int64)[0]),
                    out, ta_length)

            # concat to RaggedTensor
            return tf.nest.map_structure(
                lambda data, length: tf.RaggedTensor.from_row_lengths(
                    values=data.concat(), row_lengths=length.stack(), validate=validate
                ),
                ta_data, ta_length)

        return mapper
    return decorator
