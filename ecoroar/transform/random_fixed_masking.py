
from typing import Union

import tensorflow as tf

from ..types import TokenizedDict, InputTransform, Tokenizer


def _float_int_multiple(float_tensor, int_tensor):
    return tf.cast(float_tensor * tf.cast(int_tensor, dtype=float_tensor.dtype), dtype=int_tensor.dtype)


class RandomFixedMasking(InputTransform):
    def __init__(self, fixed_masking_ratio: Union[float, int], tokenizer: Tokenizer, seed: int = None):
        """Masks the input

        The masking procedure is:
        1. randomly mask the input with the fixed masking ratio

        Note: this algorithm uses determenstic masking, such that for fixed_masking_ratio = 90%
            exactly 90% (rounded down) of the tokens will be masked.

        Args:
            fixed_masking_ratio (Union[float, int]): The masking ratio, between 0 and 1 inclusive
            tokenizer (Tokenizer): tokenizer, specifically used to provide the mask_token_id
            seed (int, optional): Seed used to generate random masking. Defaults to None.
        """
        if not (0 <= float(fixed_masking_ratio) <= 1):
            raise TypeError(f'fixed_masking_ratio must be between 0 and 1, was "{fixed_masking_ratio}"')

        self._fixed_masking_ratio = float(fixed_masking_ratio)
        self._tokenizer = tokenizer
        self._seed = seed

        if seed is None:
            self._rng = tf.random.Generator.from_non_deterministic_state()
        else:
            self._rng = tf.random.Generator.from_seed(seed)

    @tf.function(reduce_retracing=True)
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

        if self._fixed_masking_ratio > 0:
            # e.g. input_ids = [['[BOS]', 'A', 'B', 'C', '[EOS]']]

            # True if token is allowed to be masked.
            # This prevents [BOS], [SEP], and [EOS] from being masked.
            # e.g. maskable_indicator = [[False, True, True, True, False]]
            maskable_indicator = tf.math.reduce_all(
                tf.expand_dims(input_ids, 0) != tf.reshape(self._tokenizer.kept_tokens, [-1, 1, 1]),
                axis=0
            )
            # This prevents unattended tokens from being masked, such as [PAD].
            maskable_indicator = tf.math.logical_and(
                maskable_indicator,
                tf.cast(attention_mask, tf.dtypes.bool)
            )

            # Compute the number of masked values required
            # e.g. number_of_masked_values = [2]
            sequence_length = tf.math.reduce_sum(tf.cast(maskable_indicator, dtype=tf.dtypes.int32), axis=1)
            number_of_masked_values = _float_int_multiple(
                tf.cast(self._fixed_masking_ratio, tf.dtypes.float32),
                sequence_length
            )
            total_number_of_masked_values = tf.math.reduce_sum(number_of_masked_values)

            # Sample number_of_masked_values random indices. The indices are defined by maskable_indicator.
            # masked_indices.concat() = [[0, 1], [0, 2]]
            masked_indices = tf.TensorArray(tf.int64, size=batch_size, infer_shape=False, element_shape=(None, 2))
            for obs_i in tf.range(batch_size):
                # maskable_indices = [1, 2, 3]
                maskable_indices = tf.squeeze(tf.where(maskable_indicator[obs_i, :]), axis=1)

                # Select number_of_masked_values elements from maskable_indices without replacement
                # selected_masked_indices = [1, 2]
                _, random_indices = tf.math.top_k(self._rng.uniform_full_int([sequence_length[obs_i]]),
                                                  k=number_of_masked_values[obs_i],
                                                  sorted=False)
                selected_masked_indices = tf.gather(maskable_indices, indices=random_indices)

                # Append indices to masked_indices. Note indices are sorted to provide valid
                #  ordering for SparseTensor.
                masked_indices = masked_indices.write(obs_i, tf.concat([
                    tf.fill([number_of_masked_values[obs_i], 1], tf.cast(obs_i, dtype=tf.dtypes.int64)),
                    tf.reshape(tf.sort(selected_masked_indices), [-1, 1])
                ], axis=1))

            # convert indices to tensor with true values
            # e.g. masking_indicator = [[False, True, True, False, False]]
            # NOTE: Could maybe use tf.tensor_scatter_nd_update instead
            masking_indicator = tf.sparse.SparseTensor(
                indices=masked_indices.concat(),
                values=tf.ones((total_number_of_masked_values, ), dtype=tf.dtypes.bool),
                dense_shape=(batch_size, max_sequence_length))
            masking_indicator = tf.sparse.to_dense(masking_indicator)

            # Use masking_indicator to mask the input_ids
            # e.g. input_ids = [['[BOS]', '[MASK]', '[MASK]', 'C', '[EOS]']]
            input_ids = tf.where(masking_indicator, self._tokenizer.mask_token_id, input_ids)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
