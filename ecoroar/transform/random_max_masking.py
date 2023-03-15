
import tensorflow as tf

from ..types import TokenizedDict, InputTransform


class RandomMaxMasking(InputTransform):
    def __init__(self, max_masking_ratio: float, tokenizer, seed: int = None):
        """Masks the input

        The masking procedure is:
        1. sample a masking ratio from uniform[0, max_masking_ratio)
        2. randomly mask the input with the masking ratio

        Note: this algorithm uses stocastic masking, such that for max_masking_ratio = 90%
            on average 90% of the tokens will be masked.

        Args:
            max_masking_ratio (float): The maximum masking ratio, between 0 and 1 inclusive
            tokenizer (Tokenizer): tokenizer, specifically used to provide the mask_token_id
            seed (int, optional): Seed used to generate random masking. Defaults to None.
        """
        if not (0 <= float(max_masking_ratio) <= 1):
            raise TypeError(f'max_masking_ratio must be a float between 0 and 1, was "{max_masking_ratio}"')

        self._max_masking_ratio = float(max_masking_ratio)
        self._tokenizer = tokenizer
        self._seed = seed

        if seed is None:
            self._rng = tf.random.Generator.from_non_deterministic_state()
        else:
            self._rng = tf.random.Generator.from_seed(seed)

    @tf.function
    def __call__(self, x: TokenizedDict) -> TokenizedDict:
        """Randomly mask tokenized input.

        Args:
            x (TokenizedDict): Tokenized input

        Returns:
            TokenizedDict: Masked tokenized input
        """
        input_ids = tf.ensure_shape(x['input_ids'], [None, None])
        attention_mask = tf.ensure_shape(x['attention_mask'], [None, None])
        batch_size, max_sequence_length = tf.unstack(tf.shape(input_ids), num=2)

        if self._max_masking_ratio > 0:
            # True if token is allowed to be masked.
            # This prevents [BOS], [SEP], and [EOS] from being masked.
            maskable_indicator = tf.math.reduce_all(
                tf.expand_dims(input_ids, 0) != tf.reshape(self._tokenizer.kept_tokens, [-1, 1, 1]),
                axis=0
            )

            # This prevents unattended tokens from being masked, such as [PAD].
            maskable_indicator = tf.math.logical_and(
                maskable_indicator,
                tf.cast(attention_mask, tf.dtypes.bool)
            )

            # masking ratio will be [0, max_masking_ratio)
            masking_ratio = self._rng.uniform((batch_size, 1), maxval=self._max_masking_ratio)
            masking_indicator = tf.math.logical_and(
                self._rng.uniform((batch_size, max_sequence_length)) <= masking_ratio,
                maskable_indicator
            )

            input_ids = tf.where(masking_indicator, self._tokenizer.mask_token_id, input_ids)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
