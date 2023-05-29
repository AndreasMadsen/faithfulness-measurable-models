
from dataclasses import dataclass
from typing import Union, Optional, Dict, Iterable

import tensorflow as tf

from ..types import TokenizedDict, EmbeddingDict, Model, Tokenizer

@dataclass
class SimpleOutput():
    logits: tf.Tensor
    hidden_states: Optional[tf.Tensor] = None

class LookupTestConfig():
    model_type = 'lookup test'

    def __init__(self, vocab_size) -> None:
        self.vocab_size = vocab_size

    @property
    def layers(self) -> tf.Variable:
        raise NotImplementedError('layers does not exist for LookupTestConfig')

    @property
    def num_hidden_layers(self) -> tf.Variable:
        raise NotImplementedError('num_hidden_layers does not exist for LookupTestConfig')

    @property
    def hidden_size(self) -> tf.Variable:
        raise NotImplementedError('hidden_size does not exist for LookupTestConfig')

class LookupTestModel(Model):
    def __init__(self, keys: TokenizedDict, values: tf.Tensor, vocab_size=5) -> None:
        """Maps tokenized inputs (keys) to logits (values).

        This model uses a hash lookup and is therefore not differentiable.

        Args:
            keys (TokenizedDict): Tokenized inputs
            values (tf.Tensor): Logits
            vocab_size (int, optional): The vocabulary size. Defaults to 4.
        """
        super().__init__()

        self.config = LookupTestConfig(vocab_size=vocab_size)

        self._base = tf.convert_to_tensor(self.config.vocab_size + 1, dtype=tf.dtypes.int32)
        self._logits = values
        self._keymap = self._convert_input_ids_to_int32(keys)

    @classmethod
    def from_string(cls, tokenizer: Tokenizer, mapping: Dict[str, Iterable[float]], **kwargs):
        """Constructs a LookupTestModel from a dictionary of strings to floats

        Args:
            tokenizer (Tokenizer): The tokenizer used to encode the strings
            mapping (Dict[str, Iterable[float]]): A dictionary that maps from strings to floats

        Returns:
            LookupTestModel: An LookupTestModel instance
        """
        mapping_tf = [
            (key, tf.convert_to_tensor(value, dtype=tf.dtypes.float32))
            for key, value in mapping.items()
        ]

        keys, values = tf.data.experimental.from_list(mapping_tf) \
            .map(lambda doc, logit: (tokenizer((doc, )), logit)) \
            .padded_batch(len(mapping), padding_values=(tokenizer.padding_values, None)) \
            .get_single_element()

        return cls(keys, values, **kwargs)

    def _convert_input_ids_to_int32(self, inputs: TokenizedDict) -> tf.Tensor:
        """Converts a tokenized input to an lookup id

        tf.lookup.StaticHashTable does not support looking up tuples. So the
        tokenized inputs are encoded as scalar integers, by the standard process:
            int = \\sum_i v_i * B^i

        Args:
            inputs (TokenizedDict): tokenized inputs with attention_mask

        Returns:
            tf.Tensor: sequences encoded as integers
        """
        input_ids = tf.ensure_shape(inputs['input_ids'], (None, None))
        attention_mask = tf.ensure_shape(inputs['attention_mask'], (None, None))

        _, sequence_length =  tf.unstack(tf.shape(input_ids), num=2)
        pows = self._base ** tf.range(sequence_length)
        parts = tf.where(attention_mask == 1, (input_ids + 1) * pows, 0)
        integer = tf.math.reduce_sum(parts, axis=1)
        return integer

    def _lookup(self, integer: tf.Tensor) -> tf.Tensor:
        match = tf.expand_dims(integer, 1) == tf.expand_dims(self._keymap, 0)
        tf.debugging.assert_equal(tf.math.reduce_any(match, axis=1), True)
        return tf.argmax(match, axis=1, output_type=tf.dtypes.int32)

    def call(self, inputs: Union[TokenizedDict, EmbeddingDict], training=False, output_hidden_states=False) -> SimpleOutput:
        z1 = self._convert_input_ids_to_int32(inputs)
        z2 = tf.gather(self._logits, self._lookup(z1))

        return SimpleOutput(
            logits=z2,
            hidden_states=(z1, ) if output_hidden_states else None
        )

    def inputs_embeds(self, x: TokenizedDict, training=False) -> EmbeddingDict:
        raise NotImplementedError('inputs_embeds does not exist for LookupTestModel')

    @property
    def embedding_matrix(self) -> tf.Variable:
        raise NotImplementedError('embedding_matrix does not exist for LookupTestModel')
