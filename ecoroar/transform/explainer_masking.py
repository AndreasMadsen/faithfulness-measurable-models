
from typing import Callable

import numpy as np
import tensorflow as tf

from ..types import TokenizedDict, InputTransform, Tokenizer
from .map_on_gpu import MapOnGPU
from .sequence_identifier import SequenceIndentifier

def _float_int_multiple(float_tensor, int_tensor):
    return tf.cast(float_tensor * tf.cast(int_tensor, dtype=float_tensor.dtype), dtype=int_tensor.dtype)

class ExplainerMasking(InputTransform):
    def __init__(self, explainer, tokenizer: Tokenizer, recursive: bool=True):
        """Masks the input according to the importance measure provided by an explainer

        Args:
            explainer (ImportanceMeasure): this will be called to explain the (x, y) pair
            tokenizer (Tokenizer): tokenizer, specifically used to provide the mask_token_id
            recursive (bool, optional): should the importance measure be reevaluated
        """

        self._explainer = explainer
        self._tokenizer = tokenizer
        self._recursive = recursive

        self._sequence_identifier = SequenceIndentifier(tokenizer)

    @tf.function(reduce_retracing=True)
    def _mask_inputs_ids_with_im(self, input_ids: tf.Tensor, im: tf.Tensor, masking_ratio: tf.Tensor):
        first_sequence_mask = self._sequence_identifier(input_ids) == 1

        # ensure that already masked values will continue to be masked,
        # by assigning them infinite importance.
        im = tf.where(input_ids == self._tokenizer.mask_token_id, np.inf, im)
        # ensure that kept tokens, such as [BOS] and [EOS] will remain
        im = tf.where(first_sequence_mask, im, -np.inf)

        # Rank importance measure
        ranking = tf.argsort(im, axis=1, direction='DESCENDING', stable=True)

        # Create an indice tensor mask_ranking[batch_idx] = [token_idx, ...] tensor with the
        # top `masking_ratio` elements selected. Make sure that kept_tokens are not selected.
        maskable_num_of_tokens = tf.math.reduce_sum(tf.cast(first_sequence_mask, tf.dtypes.int32), axis=1)
        mask_lengths = _float_int_multiple(masking_ratio, maskable_num_of_tokens)
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
    def _mask_input(self, x: TokenizedDict, y: tf.Tensor, im: tf.Tensor, masking_ratio: tf.Tensor) -> TokenizedDict:
        return {
            'input_ids': self._mask_inputs_ids_with_im(x['input_ids'], im, masking_ratio),
            'attention_mask': x['attention_mask']
        }

    @tf.function(reduce_retracing=True)
    def _mask_input_100p(self, x: TokenizedDict, y: tf.Tensor) -> TokenizedDict:
        # Mask 100%
        input_ids = x['input_ids']
        first_sequence_mask = self._sequence_identifier(input_ids) == 1

        return {
            'input_ids': tf.where(first_sequence_mask, self._tokenizer.mask_token_id, input_ids),
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

        masking_ratio_tf = tf.convert_to_tensor(masking_ratio, dtype=tf.dtypes.float32)

        # define mapping function
        if masking_ratio == 0.0:
            if self._recursive:
                return lambda dataset: dataset.map(
                   lambda x, y: (x, y, tf.zeros((tf.shape(x['input_ids'])[0], 0), dtype=tf.dtypes.float32))
                )
            else:
                def _mapper(x, y):
                    im = self._explainer(x, y).to_tensor(default_value=-np.inf, shape=tf.shape(x['input_ids']))
                    return (x, y, im)
                def _output_signature(dataset):
                    return (*tf.data.experimental.get_structure(dataset),
                            tf.TensorSpec(shape=(None,None), dtype=tf.dtypes.float32))

        elif masking_ratio == 1.0:
            # Avoid computing importance measure at 100% masking
            def _mapper(x, y, im):
                return (self._mask_input_100p(x, y), y, im)
            def _output_signature(dataset):
                return tf.data.experimental.get_structure(dataset)

        else:
            if self._recursive:
                def _mapper(x, y, im_empty):
                    im = self._explainer(x, y).to_tensor(default_value=-np.inf, shape=tf.shape(x['input_ids']))
                    return (self._mask_input(x, y, im, masking_ratio_tf), y, im_empty)
            else:
                def _mapper(x, y, im):
                    return (self._mask_input(x, y, im, masking_ratio_tf), y, im)

            def _output_signature(dataset):
                return tf.data.experimental.get_structure(dataset)

        return MapOnGPU(_mapper, _output_signature)
