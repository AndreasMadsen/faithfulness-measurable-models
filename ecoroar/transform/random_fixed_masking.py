
import tensorflow as tf

from ..types import TokenizedDict


class RandomFixedMasking:
    def __init__(self, fixed_masking_ratio: float, tokenizer, seed: int = None):
        """Masks the input

        The masking procedure is:
        1. randomly mask the input with the fixed masking ratio

        Note: this algorithm uses determenstic masking, such that for fixed_masking_ratio = 90%
            exactly 90% (rounded down) of the tokens will be masked.

        Args:
            fixed_masking_ratio (float): The masking ratio, between 0 and 1 inclusive
            tokenizer (Tokenizer): tokenizer, specifically used to provide the mask_token_id
            seed (int, optional): Seed used to generate random masking. Defaults to None.
        """
        if not isinstance(fixed_masking_ratio, float) or not (0 <= fixed_masking_ratio <= 1):
            raise TypeError(f'fixed_masking_ratio must be a float between 0 and 1, was "{fixed_masking_ratio}"')

        self._fixed_masking_ratio = fixed_masking_ratio
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

        if self._fixed_masking_ratio > 0:
            # e.g. input_ids = ['[BOS]', 'A', 'B', 'C', '[EOS]']

            # True if token is allowed to be masked.
            # This prevents [BOS], [SEP], and [EOS] from being masked.
            # e.g. maskable_indicator = [False, True, True, True, False]
            maskable_indicator = tf.math.reduce_all(
                tf.expand_dims(input_ids, 0) != tf.expand_dims(self._tokenizer.kept_tokens, -1),
                axis=0
            )
            # This prevents unattended tokens from being masked, such as [PAD].
            maskable_indicator = tf.math.logical_and(
                maskable_indicator,
                tf.cast(attention_mask, tf.dtypes.bool)
            )

            # Get the maskable indices
            # e.g. maskable_indices = [1, 2, 3]
            maskable_indices = tf.squeeze(tf.where(maskable_indicator), axis=1)

            # Compute the number of masked values required
            # e.g. number_of_masked_values = 2
            number_of_masked_values = tf.cast(tf.math.floor(
                self._fixed_masking_ratio * tf.cast(tf.size(maskable_indices), dtype=tf.dtypes.float32)
            ), dtype=tf.dtypes.int32)

            # Select number_of_masked_values elements from maskable_indices without replacement
            _, random_indices = tf.math.top_k(self._rng.uniform_full_int([tf.size(maskable_indices)]),
                                              k=number_of_masked_values,
                                              sorted=False)
            selected_masked_indices = tf.gather(maskable_indices, indices=random_indices)

            # convert indices to tensor with true values
            # e.g. masking_indicator = [False, True, True, False, False]
            masking_indicator = tf.sparse.SparseTensor(
                indices=tf.expand_dims(tf.sort(selected_masked_indices), axis=1),
                values=tf.ones_like(selected_masked_indices, dtype=tf.dtypes.bool),
                dense_shape=tf.shape(maskable_indicator, out_type=tf.dtypes.int64))
            masking_indicator = tf.sparse.to_dense(masking_indicator)

            # Use masking_indicator to mask the input_ids
            # e.g. input_ids = ['[BOS]', '[MASK]', '[MASK]', 'C', '[EOS]']
            input_ids = tf.where(masking_indicator, self._tokenizer.mask_token_id, input_ids)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
