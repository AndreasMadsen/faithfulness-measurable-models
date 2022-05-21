from functools import cached_property
from typing import List, Iterable
from abc import ABCMeta, abstractmethod

import tensorflow as tf
import transformers

from ..types import TokenizedDict


class AbstractTokenizer(metaclass=ABCMeta):

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

    name: str
    vocab_size: tf.Tensor

    padding_values: TokenizedDict
    vocab:  List[str]

    @abstractmethod
    def __call__(self, texts: Iterable[tf.Tensor]) -> TokenizedDict:
        ...
