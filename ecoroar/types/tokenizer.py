
from abc import ABCMeta, abstractmethod
from typing import TypedDict, Union, List, Iterable

import tensorflow as tf


class TokenizedDict(TypedDict):
    input_ids: Union[tf.Tensor, tf.RaggedTensor]
    attention_mask: Union[tf.Tensor, tf.RaggedTensor]


class EmbeddingDict(TypedDict):
    inputs_embeds: Union[tf.Tensor, tf.RaggedTensor]
    attention_mask: Union[tf.Tensor, tf.RaggedTensor]


class Tokenizer(metaclass=ABCMeta):

    unk_token: str
    unk_token_id: tf.Tensor
    pad_token: str
    pad_token_id: tf.Tensor

    bos_token: str
    bos_token_id: tf.Tensor
    sep_token: str
    sep_token_id: tf.Tensor
    eos_token: str
    eos_token_id : tf.Tensor

    mask_token: str
    mask_token_id: tf.Tensor

    kept_tokens = tf.Tensor

    alias_name: str
    model_name: str
    vocab_size: tf.Tensor

    padding_values: TokenizedDict
    vocab: List[str]

    @abstractmethod
    def __call__(self, texts: Iterable[tf.Tensor]) -> TokenizedDict:
        """Tokenizes a text tensor

        Args:
            texts (Iterable[tf.Tensor]): text tensors to be tokenized

        Returns:
            TokenizedDict: the tokenized text. Dict has
                input_ids and attention_mask properties.
        """
        ...
