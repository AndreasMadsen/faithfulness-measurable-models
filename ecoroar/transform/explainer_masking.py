
from typing import Callable

import numpy as np
import tensorflow as tf

from ..types import TokenizedDict, InputTransform, Tokenizer


class ExplainerMasking(InputTransform):
    def __init__(self, explainer, tokenizer: Tokenizer):
        """Masks the input according to the importance measure provided by an explainer

        Args:
            explainer (ImportanceMeasure): this will be called to explain the (x, y) pair
            tokenizer (Tokenizer): tokenizer, specifically used to provide the mask_token_id
        """

        self._explainer = explainer
        self._tokenizer = tokenizer

    @tf.function(reduce_retracing=True)
    def _mask_inputs_ids_with_im(self, input_ids: tf.Tensor, im: tf.RaggedTensor, masking_ratio: tf.Tensor):
        # ensure that already masked values will continue to be masked,
        # by assigning them infinite importance.
        im_dense = im.to_tensor(default_value=-np.inf, shape=tf.shape(input_ids))
        im_dense = tf.where(input_ids == self._tokenizer.mask_token_id, np.inf, im_dense)
        kept_tokens = tf.math.reduce_any(
            tf.expand_dims(input_ids, 0) == tf.reshape(self._tokenizer.kept_tokens, [-1, 1, 1]),
            axis=0
        )
        im_dense = tf.where(kept_tokens, -np.inf, im_dense)

        # Rank importance measure
        ranking = tf.argsort(im_dense, axis=1, direction='DESCENDING', stable=True)

        # Create an indice tensor [[batch_idx, token_idx], ... ] tensor with the
        # top `masking_ratio` elements selected. Make sure that kept_tokens are not
        # selected, by removing them from the sequence_length.
        sequence_length = im.row_lengths()
        maskable_num_of_tokens = sequence_length - tf.math.reduce_sum(
            tf.cast(kept_tokens, sequence_length.dtype),
            axis=1)
        mask_lengths = tf.cast(
            tf.cast(maskable_num_of_tokens, masking_ratio.dtype) * masking_ratio,
            dtype=sequence_length.dtype)
        mask_ranking = tf.RaggedTensor.from_tensor(ranking, lengths=mask_lengths)

        # Set masked elements to have the [MASK] token in the input
        return tf.tensor_scatter_nd_update(
            tensor=input_ids,
            indices=tf.stack([
                tf.cast(mask_ranking.value_rowids(), dtype=mask_ranking.flat_values.dtype),
                mask_ranking.flat_values
            ], axis=1),
            updates=tf.fill((tf.size(mask_ranking), ), self._tokenizer.mask_token_id)
        )

    @tf.function(reduce_retracing=True)
    def _mask_input(self, x: TokenizedDict, y: tf.Tensor, masking_ratio: tf.Tensor) -> TokenizedDict:
        # Compute importance measure
        im = self._explainer(x, y)

        return {
            'input_ids': self._mask_inputs_ids_with_im(x['input_ids'], im, masking_ratio),
            'attention_mask': x['attention_mask']
        }

    def __call__(self, masking_ratio: float) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
        """Mask tokens according to the explainer.

        Args:
            masking_ratio (float): The ratio of tokens to mask according to the explainer,
                between 0 and 1 inclusive

        Returns:
            Callable[[tf.data.Dataset], tf.data.Dataset]: function that maps from dataset to dataset
        """
        if not (0 <= float(masking_ratio) <= 1):
            raise TypeError(f'masking_ratio must be between 0 and 1, was "{masking_ratio}"')

        masking_ratio = tf.convert_to_tensor(masking_ratio, dtype=tf.dtypes.float32)

        def masked_dataset(dataset):
            if masking_ratio == 0.0:
                return dataset

            # tf.Dataset.from_generator is used because tf.Dataset.map will run on the CPU,
            # and the explainer is likely to do model inference and perhaps derivatives.
            # Those are expensive calculations, and should be done on the GPU. using
            # `with tf.device('GPU')` to force calculations on the GPU is an option too.
            # However, this is still 2x-3x slower, likely because data is not prefetched to
            # the GPU.
            def _generator():
                for x, y in dataset.prefetch(tf.data.AUTOTUNE):
                    yield (self._mask_input(x, y, masking_ratio), y)

            return tf.data.Dataset.from_generator(
                _generator,
                output_signature=tf.data.experimental.get_structure(dataset)
            ).apply(tf.data.experimental.assert_cardinality(dataset.cardinality()))

        return masked_dataset
