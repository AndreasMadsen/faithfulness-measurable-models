
import tensorflow as tf

from ._importance_measure import ImportanceMeasure


class LeaveOneOutSign(ImportanceMeasure):
    _implements_explain_batch = False
    _name = 'loo-sign'

    def _explain_observation(self, x, y):
        sequence_length = tf.shape(x['input_ids'])[0]

        # TODO: for masked inputs, [MASK] would be replaced with [MASK].
        #       This enforces zero attribution score. Therefore this be optimized by
        #       skipping the model evaluation.
        unmaskable_tokens_ids = tf.concat((self._tokenizer.kept_tokens, [self._tokenizer.mask_token_id]), axis=0)
        maskable_tokens = tf.math.reduce_all(
            tf.expand_dims(x['input_ids'], 0) != tf.expand_dims(unmaskable_tokens_ids, 1),
            axis=0
        )
        token_idx_to_mask = tf.cast(tf.squeeze(tf.where(maskable_tokens), axis=1), dtype=tf.dtypes.int32)
        n_samples = tf.size(token_idx_to_mask)

        # x_masked[-1] provides the baseline
        x_masked = tf.nest.map_structure(
            lambda item: tf.repeat(tf.expand_dims(item, 0), n_samples + 1, axis=0),
            x)
        x_masked['input_ids'] = tf.tensor_scatter_nd_update(
            x_masked['input_ids'],
            tf.stack([tf.range(n_samples), token_idx_to_mask], axis=1),
            tf.fill(n_samples, self._tokenizer.mask_token_id)
        )

        # batch evaluate the masked examples in x_masked
        output_dtype = tf.keras.mixed_precision.global_policy().compute_dtype
        num_of_batches = tf.cast(tf.math.ceil((n_samples + 1) / self._inference_batch_size), dtype=tf.dtypes.int32)
        predict_all_array = tf.TensorArray(output_dtype, size=num_of_batches, infer_shape=False, element_shape=(None, ))
        for batch_i in tf.range(num_of_batches):
            x_batch = tf.nest.map_structure(
                lambda item: item[batch_i*self._inference_batch_size:(batch_i + 1)*self._inference_batch_size, ...],
                x_masked)

            y_batch = self._model(x_batch).logits[:, y]
            predict_all_array = predict_all_array.write(batch_i, y_batch)
        predicts_all = predict_all_array.concat()

        # Compute predictive impact relative to baseline
        importance = predicts_all[:-1] - predicts_all[-1:]

        # Reshape importance to the sequence_length, fill blanks with -inf
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
