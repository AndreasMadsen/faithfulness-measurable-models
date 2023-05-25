
from typing import Any
import tensorflow as tf

from ..types import Model
from ..util import get_compiler
from ..types import TokenizedDict

@tf.function
def _log2int(x):
    log2 = tf.math.log(tf.cast(2, dtype=tf.dtypes.float32))
    x_float = tf.cast(x, dtype=tf.dtypes.float32)
    return tf.cast(tf.math.log(x_float) / log2, dtype=x.dtype)

class BatchEvaluator:
    def __init__(self, model: Model, batch_size: tf.Tensor, run_eagerly: bool = False, jit_compile: bool = False) -> None:
        self._model = model
        self._batch_size = batch_size

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

    def _jit_batcher(self, x: TokenizedDict, y: tf.Tensor) -> tf.Tensor:
        # batch evaluate the masked examples in x_masked
        output_dtype = tf.keras.mixed_precision.global_policy().compute_dtype
        num_of_samples = tf.shape(x['input_ids'])[0]

        # This chucks the input into batches of self._batch_size (B) size.
        #   This is fine for the JIT. However, for the remainder which may not have size B,
        #   this is a problem because the JIT will have to recompile for every batch size between 1 and B.
        #   Instead, this uses a step down approach. This changes the JIT complications from $n$ to $log2(n)$.
        #   For example:
        #     16, 16, 8, 4, 2, 1. This would reduce from 16*5=80 to 5*5=25 JIT compilations.
        predict_all = tf.zeros((num_of_samples, ), dtype=output_dtype)
        batch_start = 0
        while batch_start < num_of_samples:
            batch_size = tf.minimum(num_of_samples - batch_start, self._batch_size)
            # If the batch_size does not divide (self._batch_size) which is assumed to be a power of two,
            #   then round down the batch size to the highest power of two that is viable.
            batch_size = tf.math.pow(2, _log2int(batch_size))

            # Extract and evaluate batch
            batch_end = batch_start + batch_size
            x_batch = tf.nest.map_structure(lambda item: item[batch_start:batch_end, ...], x)
            y_batch = y[batch_start:batch_end]
            predict_batch = self._wrap_logits(x_batch, y_batch)

            # Infill results
            predict_all = tf.tensor_scatter_nd_update(
                predict_all,
                tf.expand_dims(tf.range(batch_start, batch_end), axis=1),
                predict_batch
            )

            # prepear for next iteration
            batch_start = batch_end

        return predict_all

    def _std_batcher(self, x: TokenizedDict, y: tf.Tensor) -> tf.Tensor:
        # batch evaluate the masked examples in x_masked
        output_dtype = tf.keras.mixed_precision.global_policy().compute_dtype
        num_of_samples = tf.shape(x['input_ids'])[0]
        num_of_batches = tf.cast(tf.math.ceil(num_of_samples / self._batch_size), dtype=tf.dtypes.int32)

        predict_all_array = tf.TensorArray(output_dtype, size=num_of_batches, infer_shape=False, element_shape=(None, ))
        for batch_i in tf.range(num_of_batches):
            batch_start = batch_i*self._batch_size
            batch_end = tf.minimum(num_of_samples, (batch_i + 1)*self._batch_size)

            x_batch = tf.nest.map_structure(lambda item: item[batch_start:batch_end, ...], x)
            y_batch = y[batch_start:batch_end]
            predict_batch = self._wrap_logits(x_batch, y_batch)

            predict_all_array = predict_all_array.write(batch_i, predict_batch)
        return predict_all_array.concat()

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
