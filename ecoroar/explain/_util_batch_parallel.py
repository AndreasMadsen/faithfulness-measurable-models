
import tensorflow as tf

def batch_parallel(batch_size: int, num_parallel_calls=tf.data.AUTOTUNE, validate=True):
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
            with tf.name_scope('batch_parallel'):
                @tf.function(reduce_retracing=True)
                def wrap_fn(args):
                    with tf.name_scope('obs_eval'):
                        return fn(*args, *extra_args)

                structured_outputs = wrap_fn.get_concrete_function(
                    tf.nest.map_structure(lambda item: item[0], batch_args)
                ).structured_outputs

                return tf.map_fn(
                    wrap_fn, batch_args,
                    parallel_iterations=num_parallel_calls, infer_shape=False,
                    fn_output_signature=tf.nest.map_structure(
                        lambda item: tf.RaggedTensorSpec(
                            shape=[None] + item.shape[1:],
                            dtype=item.dtype,
                            ragged_rank=0
                        ),
                        structured_outputs)
                )

        return mapper
    return decorator
