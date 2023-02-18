
import tensorflow as tf
from ..types import Tokenizer, TokenizedDict

class SimpleTestTokenizer(Tokenizer):
    def __init__(self):
        self.unk_token = '[UNK]'
        self.unk_token_id = tf.constant(-1, tf.dtypes.int32)
        self.pad_token = '[PAD]'
        self.pad_token_id = tf.constant(2, tf.dtypes.int32)

        self.bos_token = '[BOS]'
        self.bos_token_id =tf.constant(0, tf.dtypes.int32)
        self.eos_token = '[EOS]'
        self.eos_token_id = tf.constant(1, tf.dtypes.int32)

        self.kept_tokens = tf.stack([
            self.bos_token_id,
            self.eos_token_id
        ])

        self.vocab_size = tf.constant(4, tf.dtypes.int32)
        self.padding_values = {
            'input_ids': self.pad_token_id,
            'attention_mask': tf.constant(0, dtype=tf.dtypes.int8)
        }
        self.vocab = [
            self.bos_token,
            self.eos_token,
            self.pad_token,
            'token'
        ]
        self._token_to_id = { token: i for i, token in enumerate(self.vocab) }

    def _wrap_tokenizer_call(self, texts):
        input_ids = []
        attention_mask = []
        for text in texts:
            tokens = text.numpy().decode('utf-8').split(' ')
            input_ids += [self._token_to_id[token] for token in tokens]
            attention_mask += [int(token != self.pad_token) for token in tokens]

        return (input_ids, attention_mask)

    @tf.function
    def __call__(self, texts: tf.Tensor) -> TokenizedDict:
        input_ids, attention_mask = tf.py_function(
            self._wrap_tokenizer_call,
            inp=[texts],
            Tout=[tf.dtypes.int32, tf.dtypes.int8]
        )

        return {
            'input_ids': tf.ensure_shape(input_ids, [None]),
            'attention_mask': tf.ensure_shape(attention_mask, [None])
        }
