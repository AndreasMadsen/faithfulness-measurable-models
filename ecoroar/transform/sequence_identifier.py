
import tensorflow as tf

from ..types import TokenizedDict, InputTransform, Tokenizer


class SequenceIndentifier(InputTransform):
    def __init__(self, tokenizer: Tokenizer):
        """Annotates which sequence each input token is assigned to

        0: neither sequence ([PAD] | [BOS] | [EOS] | [SEP] tokens)
        1: the first sequence
        2: the second sequence

        Args:
            tokenizer (Tokenizer): tokenizer, specifically used to provide the special token ids
        """

        self._tokenizer = tokenizer

    def __call__(self, x: TokenizedDict) -> tf.Tensor:
        """Annotates which sequence each input token is assigned to

        Args:
            x (TokenizedDict): Tokenized input

        Returns:
            tf.Tensor(dtype=tf.int8): same shape as x['inputs_ids']
        """

        input_ids = tf.ensure_shape(x['input_ids'], [None, None])
        batch_size, _ = tf.unstack(tf.shape(input_ids), num=2)

        # is_sequence_token         = [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0]
        is_sequence_token = tf.math.reduce_all(
            tf.expand_dims(input_ids, 2) != self._tokenizer.kept_tokens[None, None, :],
            axis=-1
        )

        # is_sequence_token[:, 1:]  = [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0]
        # is_sequence_token[:, :-1] = [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0]
        # sequence_ended            = [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
        sequence_ended = tf.math.logical_and(
            tf.math.logical_not(is_sequence_token[:, 1:]),
            is_sequence_token[:, :-1]
        )

        # sequence_ended_1pad       = [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
        sequence_ended_1pad = tf.concat((
            tf.ones((batch_size, 1), dtype=sequence_ended.dtype),
            sequence_ended
        ), axis=1)

        # identifiers               =  [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3]
        # NOTE: Using int32 here because XLA does not support int8 or int16
        identifiers = tf.cast(tf.cumsum(tf.cast(sequence_ended_1pad, dtype=tf.dtypes.int32), axis=1), dtype=tf.dtypes.int8)

        # identifiers_masked        =  [0, 1, 1, 1, 0, 0, 2, 2, 2, 0, 0, 0]
        identifiers_masked = tf.where(is_sequence_token, identifiers, tf.constant(0, dtype=identifiers.dtype))

        return identifiers_masked
