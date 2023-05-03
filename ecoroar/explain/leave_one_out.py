
import tensorflow as tf

from ecoroar.types import Tokenizer

from ._importance_measure import ImportanceMeasureObservation
from ..transform.sequence_identifier import SequenceIndentifier

class LeaveOneOutSign(ImportanceMeasureObservation):
    _name = 'loo-sign'
    def __init__(self, tokenizer: Tokenizer, *args, **kwargs) -> None:
        super().__init__(tokenizer, *args, **kwargs)
        self._sequence_identifier = SequenceIndentifier(tokenizer)

    def _explain_observation(self, x, y):
        input_ids = x['input_ids']
        sequence_length = tf.shape(input_ids)[0]

        # Since only explanation w.r.t. the first sequence are considred, only attempt
        # LOO measures on the first sequence.
        maskable_tokens = tf.squeeze(self._sequence_identifier(tf.expand_dims(input_ids, 0)), 0) == 1
        # For masked inputs, [MASK] would be replaced with [MASK].
        # This enforces zero attribution score. Therefore this be optimized by
        #   skipping the model evaluation.
        maskable_tokens = tf.logical_and(maskable_tokens, input_ids != self._tokenizer.mask_token_id)

        # Identify which tokens should be probed
        token_idx_to_mask = tf.squeeze(tf.where(maskable_tokens), axis=1)
        n_samples = tf.size(token_idx_to_mask)

        # x_masked[-1] provides the baseline
        x_masked = tf.nest.map_structure(
            lambda item: tf.repeat(tf.expand_dims(item, 0), n_samples + 1, axis=0),
            x)

        x_masked['input_ids'] = tf.tensor_scatter_nd_update(
            x_masked['input_ids'],
            tf.stack([tf.range(n_samples, dtype=token_idx_to_mask.dtype), token_idx_to_mask], axis=1),
            tf.fill((n_samples, ), self._tokenizer.mask_token_id)
        )

        # batch evaluate the masked examples in x_masked
        output_dtype = tf.keras.mixed_precision.global_policy().compute_dtype
        num_of_batches = tf.cast(tf.math.ceil((n_samples + 1) / self._inference_batch_size), dtype=tf.dtypes.int32)
        predict_all_array = tf.TensorArray(output_dtype, size=num_of_batches, infer_shape=False, element_shape=(None, ))
        for batch_i in tf.range(num_of_batches):
            # NOTE: The tf.minimum should not be required, but there is a bug in XLA.
            # See: https://github.com/tensorflow/tensorflow/issues/60472
            x_batch = tf.nest.map_structure(
                lambda item: item[batch_i*self._inference_batch_size:tf.minimum(tf.shape(item)[0], (batch_i + 1)*self._inference_batch_size), ...],
                x_masked)

            y_batch = self._model(x_batch).logits[:, y]
            predict_all_array = predict_all_array.write(batch_i, y_batch)
        predicts_all = predict_all_array.concat()

        # Compute predictive impact relative to baseline
        importance = predicts_all[:n_samples] - predicts_all[n_samples]

        # Reshape importance to the sequence_length, fill blanks with 0.
        return tf.tensor_scatter_nd_update(
            tf.zeros((sequence_length, ), dtype=importance.dtype),
            tf.expand_dims(token_idx_to_mask, 1),
            importance
        )


class LeaveOneOutAbs(LeaveOneOutSign):
    _implements_explain_batch = False
    _name = 'loo-abs'

    def _explain_observation(self, x, y):
        importance = super()._explain_observation(x, y)
        return tf.math.abs(importance)
