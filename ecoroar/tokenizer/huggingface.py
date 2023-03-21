import pathlib
from functools import cached_property
from typing import List, Iterable

import tensorflow as tf
import transformers

from ..types import TokenizedDict, Tokenizer

# To save disk space and preprocessing time, map different model_names to the same tokenizer
# when the tokenizer is known to be the same.
_ALIAS_MODEL_NAME = {
    'roberta-base': 'roberta-base',
    'roberta-large': 'roberta-base',
    'andreasmadsen/efficient_mlm_m0.15': 'roberta-base',
    'andreasmadsen/efficient_mlm_m0.20': 'roberta-base',
    'andreasmadsen/efficient_mlm_m0.30': 'roberta-base',
    'andreasmadsen/efficient_mlm_m0.40': 'roberta-base',
    'andreasmadsen/efficient_mlm_m0.50': 'roberta-base',
    'andreasmadsen/efficient_mlm_m0.60': 'roberta-base',
    'andreasmadsen/efficient_mlm_m0.70': 'roberta-base',
    'andreasmadsen/efficient_mlm_m0.80': 'roberta-base',
}


class HuggingfaceTokenizer(Tokenizer):
    def __init__(self, model_name: str, persistent_dir: pathlib.Path):
        """Wrapper for a huggingface tokenizer, retrieved via transformers.AutoTokenizer

        Args:
            model_name (str): the model name as input to transformers.AutoTokenizer
            persistent_dir (pathlib.Path): used to store the downloaded tokenizer
        """
        self._model_name = model_name
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.alias_name,
            cache_dir=f'{persistent_dir}/cache/tokenizer',
            use_fast=True
        )

        self.unk_token = self._tokenizer.unk_token
        self.unk_token_id = tf.constant(self._tokenizer.unk_token_id, tf.dtypes.int32)
        self.pad_token = self._tokenizer.pad_token
        self.pad_token_id = tf.constant(self._tokenizer.pad_token_id, tf.dtypes.int32)

        self.bos_token = self._tokenizer.bos_token
        self.bos_token_id = tf.constant(self._tokenizer.bos_token_id, tf.dtypes.int32)
        self.sep_token = self._tokenizer.sep_token
        self.sep_token_id = tf.constant(self._tokenizer.sep_token_id, tf.dtypes.int32)
        self.eos_token = self._tokenizer.eos_token
        self.eos_token_id = tf.constant(self._tokenizer.eos_token_id, tf.dtypes.int32)

        self.mask_token = self._tokenizer.mask_token
        self.mask_token_id = tf.constant(self._tokenizer.mask_token_id, tf.dtypes.int32)

        self.kept_tokens = tf.constant([
            self._tokenizer.bos_token_id,
            self._tokenizer.sep_token_id,
            self._tokenizer.eos_token_id
        ], tf.dtypes.int32)

    @property
    def alias_name(self) -> str:
        if self._model_name in _ALIAS_MODEL_NAME:
            return _ALIAS_MODEL_NAME[self._model_name]
        else:
            return self._model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @cached_property
    def vocab_size(self) -> tf.Tensor:
        """Vocabulary size
        """
        return tf.constant(len(self._tokenizer.vocab), tf.dtypes.int32)

    @cached_property
    def padding_values(self) -> TokenizedDict:
        """Padding values to use

        This is useful in the context of tf.data.Dataset.padded_batch. For example:

            dataset.
                .padding_values(batch_size,
                                padding_values=(tokenizer.padding_values, None))
        """
        return {
            'input_ids': self.pad_token_id,
            'attention_mask': tf.constant(0, dtype=tf.dtypes.int8)
        }

    @cached_property
    def padding_shapes(self) -> TokenizedDict:
        """Padding shapes to use

        This is useful in the context of tf.data.Dataset.padded_batch. For example:

            dataset.
                .padding_values(batch_size,
                                padded_shapes=(tokenizer.padding_values, []))
        """
        return {
            'input_ids': [self._tokenizer.model_max_length],
            'attention_mask': [self._tokenizer.model_max_length]
        }

    @property
    def vocab(self) -> List[str]:
        """Return vocabulary
        """
        return list(self._vocabulary)

    def _wrap_tokenizer_call(self, texts: Iterable[tf.Tensor]):
        texts = tuple(text.numpy().decode('utf-8') for text in texts)
        output = self._tokenizer(*texts, truncation=True)
        return (output['input_ids'], output['attention_mask'])

    @tf.function
    def __call__(self, texts: Iterable[tf.Tensor]) -> TokenizedDict:
        """Tokenizes, encodes, and joins text, then annotates with attention_mask

        Args:
            texts (Iterable[tf.Tensor]): Tuple of text tensor. Each is a scalar.

        Returns:
            TokenizedDict: Dict of input_ids and attention_mask.
                Both have shape [sequence_length].
        """
        input_ids, attention_mask = tf.py_function(
            self._wrap_tokenizer_call,
            inp=[texts],
            Tout=[tf.dtypes.int32, tf.dtypes.int8]
        )

        return {
            'input_ids': tf.ensure_shape(input_ids, [None]),
            'attention_mask': tf.ensure_shape(attention_mask, [None])
        }
