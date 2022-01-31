
from typing import TypedDict, Union

import tensorflow as tf


class TokenizedDict(TypedDict):
    input_ids: Union[tf.Tensor, tf.RaggedTensor]
    token_type_ids: Union[tf.Tensor, tf.RaggedTensor]
    attention_mask: Union[tf.Tensor, tf.RaggedTensor]
