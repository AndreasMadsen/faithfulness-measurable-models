from functools import cached_property
from typing import List

import tensorflow as tf
import transformers

from ..types import TokenizedDict


class HuggingfaceTokenizer:
    def __init__(self, model_name: str, persistent_dir: str):
        """Wrapper for a huggingface tokenizer, retrived via transformers.AutoTokenizer

        Args:
            model_name (str): the model name as input to transformers.AutoTokenizer
            persistent_dir (str): used to store the downloaded tokenizer
        """
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=f'{persistent_dir}/download/tokenizer',
            use_fast=True
        )

        self.unk_token = self._tokenizer.unk_token
        self.unk_token_id = tf.constant(self._tokenizer.unk_token_id, tf.dtypes.int32)
        self.sep_token = self._tokenizer.sep_token
        self.sep_token_id = tf.constant(self._tokenizer.sep_token_id, tf.dtypes.int32)
        self.pad_token = self._tokenizer.pad_token
        self.pad_token_id = tf.constant(self._tokenizer.pad_token_id, tf.dtypes.int32)
        self.cls_token = self._tokenizer.cls_token
        self.cls_token_id = tf.constant(self._tokenizer.cls_token_id, tf.dtypes.int32)
        self.mask_token = self._tokenizer.mask_token
        self.mask_token_id = tf.constant(self._tokenizer.mask_token_id, tf.dtypes.int32)

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
            'attention_mask': tf.cast(self.pad_token_id, dtype=tf.dtypes.int8)
        }

    @property
    def vocab(self) -> List[str]:
        """Return vocabulary
        """
        return list(self._vocabulary)

    def _wrap_tokenizer_call(self, text):
        encode = self._tokenizer(text.numpy().decode('utf-8'), truncation=True)
        return (encode['input_ids'], encode['attention_mask'])

    @tf.function
    def __call__(self, text: tf.Tensor) -> TokenizedDict:
        """Tokenizes and encodes text, then annotates with the type and mask

        Args:
            text (tf.Tensor): text tensor, can be batched

        Returns:
            TokenizedDict: Dict of input_ids, token_type_ids, and attention_mask
        """
        input_ids, attention_mask = tf.py_function(
            self._wrap_tokenizer_call,
            inp=[text],
            Tout=[tf.dtypes.int32, tf.dtypes.int8]
        )

        return {
            'input_ids': tf.ensure_shape(input_ids, [None]),
            'attention_mask': tf.ensure_shape(attention_mask, [None])
        }
