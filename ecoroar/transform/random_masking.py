
import tensorflow as tf

from ..types import TokenizedDict


class RandomMasking:
    def __init__(self, max_masking_ratio: float, tokenizer, seed: int = None):
        """Masks the input

        The masking procedure is:
        1. sample a masking ratio from uniform[0, max_masking_ratio)
        2. randomly mask the input with the masking ratio

        Args:
            max_masking_ratio (float): The maximum masking ratio, between 0 and 1 inclusive
            tokenizer (Tokenizer): tokenizer, specifically used to provide the mask_token_id
            seed (int, optional): Seed used to generate random masking. Defaults to None.
        """
        if not isinstance(max_masking_ratio, float) or not (0 <= max_masking_ratio <= 1):
            raise TypeError(f'max_masking_ratio must be a float between 0 and 1, was "{max_masking_ratio}"')

        self._max_masking_ratio = max_masking_ratio
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
        input_ids = tf.ensure_shape(x['input_ids'], [None])
        attention_mask = tf.ensure_shape(x['attention_mask'], [None])

        if self._max_masking_ratio > 0:
            # masking ratio will be [0, max_masking_ratio)
            masking_ratio = self._rng.uniform([1], maxval=self._max_masking_ratio)
            masking_indicator = tf.math.logical_and(
                self._rng.uniform(tf.shape(input_ids)) <= masking_ratio,
                tf.cast(attention_mask, tf.dtypes.bool)
            )
            input_ids = tf.where(masking_indicator, self._tokenizer.mask_token_id, input_ids)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
