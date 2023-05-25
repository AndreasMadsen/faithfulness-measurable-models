
import tensorflow as tf

from ..types import Tokenizer, TokenizedDict
from ..transform import SequenceIndentifier

from ._importance_measure import ImportanceMeasureBatch
from ._util_evaluate import BatchEvaluator

def _batch_parallel(fn):
    @tf.function(reduce_retracing=True)
    def mapper(*args):
        return tf.data.Dataset.from_tensor_slices(args) \
            .map(fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True) \
            .ragged_batch(16) \
            .get_single_element()

    return mapper

@_batch_parallel
def _create_mask(maskable_tokens):
    # Identify which tokens should be probed
    token_idx_to_mask = tf.squeeze(tf.where(maskable_tokens), axis=1)
    n_samples = tf.size(token_idx_to_mask)

    mask_patterns = tf.tensor_scatter_nd_update(
        tf.repeat(tf.expand_dims(maskable_tokens, 0), (n_samples + 1), axis=0),
        tf.stack([tf.range(n_samples, dtype=token_idx_to_mask.dtype), token_idx_to_mask], axis=1),
        tf.zeros((n_samples, ), dtype=tf.dtypes.bool)
    )

    return (mask_patterns, token_idx_to_mask)

def _normalize_mask_patterns_shape(mask, max_sequence_length):
    # tensorflow struggle to infer the fixed-size of of the RaggedTensor, and thinks
    # that the sequence_length (third dimention) is ragged too. It is not, so do
    # the cheapest conversion back and fouth (to tensor, to ragged).
    return tf.RaggedTensor.from_row_splits(
        values=tf.reshape(mask.flat_values, [-1, max_sequence_length]),
        row_splits=mask.row_splits
    )

class LeaveOneOutSign(ImportanceMeasureBatch):
    _name = 'loo-sign'
    _defer_jit = True

    def __init__(self, tokenizer: Tokenizer, *args,
                 run_eagerly: bool = False, jit_compile: bool = False,
                 **kwargs) -> None:
        super().__init__(tokenizer, *args, run_eagerly=run_eagerly, jit_compile=jit_compile, **kwargs)
        self._sequence_identifier = SequenceIndentifier(tokenizer)
        self._evaluate = BatchEvaluator(self._model, batch_size=self._inference_batch_size,
                                        run_eagerly=run_eagerly, jit_compile=jit_compile)

    def _create_masked_inputs(self, x: TokenizedDict, mask_pattern: tf.RaggedTensor, maskable_tokens: tf.Tensor) -> TokenizedDict:
        x_repeat = tf.nest.map_structure(
            lambda item: tf.gather(item, mask_pattern.value_rowids()),
            x)

        x_repeat['input_ids'] = tf.where(
            tf.logical_and(tf.expand_dims(maskable_tokens, 1), tf.logical_not(mask_pattern)).merge_dims(0, 1),
            self._tokenizer.mask_token_id,
            x_repeat['input_ids'])

        return x_repeat

    def _explain_batch(self, x, y):
        input_ids = x['input_ids']
        batch_size, max_sequence_length = tf.unstack(tf.shape(input_ids), num=2)

        # Since only explanation w.r.t. the first sequence are considred, only attempt
        # LOO measures on the first sequence.
        maskable_tokens = self._sequence_identifier(input_ids) == 1
        # For masked inputs, [MASK] would be replaced with [MASK].
        # This enforces zero attribution score. Therefore this can  be optimized by
        #   skipping the model evaluation.
        maskable_tokens = tf.logical_and(maskable_tokens, input_ids != self._tokenizer.mask_token_id)

        # create masked inputs
        mask_patterns, mask_indices = _create_mask(maskable_tokens)
        mask_patterns = _normalize_mask_patterns_shape(mask_patterns, max_sequence_length)
        x_masked_flattened = self._create_masked_inputs(x, mask_patterns, maskable_tokens)

        # batch evaluate the masked examples in x_masked
        predict_all = tf.RaggedTensor.from_row_splits(
            values=self._evaluate(x_masked_flattened, tf.gather(y, mask_patterns.value_rowids())),
            row_splits=mask_patterns.row_splits
        )

        # Compute predictive impact relative to baseline
        importance = tf.expand_dims(predict_all[:, -1:].merge_dims(0, 1), axis=1) - predict_all[:, :-1]

        # Reshape importance to the sequence_length, fill blanks with 0.
        return tf.tensor_scatter_nd_update(
            tf.zeros((batch_size, max_sequence_length), dtype=importance.dtype),
            tf.stack([mask_indices.value_rowids(), mask_indices.merge_dims(0, 1)], axis=1),
            importance.merge_dims(0, 1)
        )


class LeaveOneOutAbs(LeaveOneOutSign):
    _name = 'loo-abs'

    def _explain_batch(self, x, y):
        importance = super()._explain_batch(x, y)
        return tf.math.abs(importance)
