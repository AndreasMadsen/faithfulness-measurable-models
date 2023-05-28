
from typing import Tuple
import tensorflow as tf

from ..types import Model
from ..util import get_compiler
from ..types import TokenizedDict

@tf.function
def _log2int(x):
    log2 = tf.math.log(tf.cast(2, dtype=tf.dtypes.float32))
    x_float = tf.cast(x, dtype=tf.dtypes.float32)
    return tf.cast(tf.math.log(x_float) / log2, dtype=x.dtype)

def _create_sub_sizes(batch_size):
    sub_batch_sizes = []
    remain = batch_size
    while remain > 0:
        remain = remain // 2
        sub_batch_sizes.append(remain)
    return tf.stack(sub_batch_sizes)

class BatchEvaluator:
    def __init__(self, model: Model, batch_size: tf.Tensor,
                 run_eagerly: bool = False, jit_compile: bool = False,
                 num_parallel_calls: int=10) -> None:
        self._model = model
        self._batch_size = batch_size
        self._batch_sub_size = _create_sub_sizes(batch_size)
        self._num_parallel_calls = num_parallel_calls

        jit_compiler = get_compiler(run_eagerly, jit_compile)
        std_compiler = get_compiler(run_eagerly, False)
        self._wrap_logits = jit_compiler(self._logits)

        if jit_compile:
            # jit_batcher will invoke more _logits calls but at fewer different shapes.
            #   This is to avoid compiling too many times.
            self._wrap_batcher = std_compiler(self._jit_batcher)
        else:
            self._wrap_batcher = std_compiler(self._std_batcher)

    def _logits(self, x_batch: TokenizedDict, y_batch: tf.Tensor) -> tf.Tensor:
        return tf.gather(self._model(x_batch).logits, y_batch, batch_dims=1)

    def _loop_batch_evaluate(self, x: TokenizedDict, y: tf.Tensor, batch_starts: tf.Tensor, batch_ends: tf.Tensor) -> tf.Tensor:
        output_dtype = tf.keras.mixed_precision.global_policy().compute_dtype

        num_of_batches = tf.size(batch_starts)
        results_ta = tf.TensorArray(output_dtype, size=num_of_batches, infer_shape=False, element_shape=(None, ))
        for batch_i in tf.range(num_of_batches):
            tf.autograph.experimental.set_loop_options(
                parallel_iterations=self._num_parallel_calls
            )

            batch_start = batch_starts[batch_i]
            batch_end = batch_ends[batch_i]
            with tf.name_scope('model_call'):
                batch_result = self._wrap_logits(
                    tf.nest.map_structure(lambda item: item[batch_start:batch_end, ...], x),
                    y[batch_start:batch_end]
                )
            results_ta = results_ta.write(batch_i, batch_result)

        return results_ta.concat()

    def _jit_batcher_range(self, num_of_samples: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        num_of_big_batches = num_of_samples // self._batch_size
        big_batches = tf.repeat(self._batch_size, num_of_big_batches)

        # This chucks the input into batches of self._batch_size (B) size.
        #   This is fine for the JIT. However, for the remainder which may not have size B,
        #   this is a problem because the JIT will have to recompile for every batch size between 1 and B.
        #   Instead, this uses a step down approach. This changes the JIT complications from $n$ to $log2(n)$.
        #   For example:
        #     16, 16, 8, 4, 2, 1. This would reduce from 16*5=80 to 5*5=25 JIT compilations.
        remaining = num_of_samples - num_of_big_batches * self._batch_size
        small_batches_select = tf.bitwise.bitwise_and(remaining, self._batch_sub_size) > 0
        small_batches = tf.gather(self._batch_sub_size, tf.squeeze(tf.where(small_batches_select), 1))

        batch_sizes = tf.concat([big_batches, small_batches], axis=0)
        batch_ends = tf.math.cumsum(batch_sizes)
        batch_starts = tf.concat([[0], batch_ends[:-1]], axis=0)

        return (batch_starts, batch_ends)

    def _jit_batcher(self, x: TokenizedDict, y: tf.Tensor) -> tf.Tensor:
        num_of_samples = tf.shape(x['input_ids'])[0]
        with tf.name_scope('jit_range'):
            batch_starts, batch_ends = self._jit_batcher_range(num_of_samples)
        with tf.name_scope('jit_eval_loop'):
            return self._loop_batch_evaluate(x, y, batch_starts, batch_ends)

    def _std_batcher_range(self, num_of_samples: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        num_of_batches = tf.cast(tf.math.ceil(num_of_samples / self._batch_size), dtype=tf.dtypes.int32)

        batch_starts = tf.range(num_of_batches) * self._batch_size
        batch_ends = tf.minimum(num_of_samples, batch_starts + self._batch_size)
        return (batch_starts, batch_ends)

    def _std_batcher(self, x: TokenizedDict, y: tf.Tensor) -> tf.Tensor:
        num_of_samples = tf.shape(x['input_ids'])[0]
        with tf.name_scope('std_range'):
            batch_starts, batch_ends = self._std_batcher_range(num_of_samples)
        with tf.name_scope('std_eval_loop'):
            return self._loop_batch_evaluate(x, y, batch_starts, batch_ends)

    def __call__(self, x: TokenizedDict, y: tf.Tensor) -> tf.Tensor:
        """Evaluates the model using the input x, and extracts the logits for class y

        This uses an internal mini batch system.

        Args:
            x (TokenizedDict): Structure of batched tensors
            y (tf.Tensor): scalar, the column index to extract

        Returns:
            tf.Tensor: vector of the output,
        """
        return self._wrap_batcher(x, y)

    def single(self, x: TokenizedDict, y: tf.Tensor) -> tf.Tensor:
        """Evaluates the model using the input x, and extracts the logits for class y

        This does not perform any batching. You could just call the model directly,
        but that would invoke another jit compilation.

        Args:
            x (TokenizedDict): Structure of batched tensors
            y (tf.Tensor): scalar, the column index to extract

        Returns:
            tf.Tensor: vector of the output,
        """
        return self._wrap_logits(x, y)
