from functools import cached_property
from typing import List, Union

import tensorflow as tf
import tensorflow_text as tftx
import transformers

from ..types import TokenizedDict


class BertTokenizer:
    def __init__(self, model_name: str, persistent_dir: str):
        """Creates a tensorflow tokenizer compatible with transformers.FastBertTokenizer

        The behavior of this is similar to transformers.FastBertTokenizer. However,
        it uses tensorflow native calls to tokenize to avoid crossing the python-C
        language boundary, i.e. it is faster.

        Args:
            model_name (str): The model name as provided to transformers.FastBertTokenizer
            persistent_dir (str): Persistent directory where the tokenizer will be downloaded to
        """
        ref = transformers.BertTokenizer.from_pretrained(
            model_name,
            cache_dir=f'{persistent_dir}/download/tokenizer')

        self.unk_token = ref.unk_token
        self.unk_token_id = tf.constant(ref.unk_token_id, tf.dtypes.int32)
        self.sep_token = ref.sep_token
        self.sep_token_id = tf.constant(ref.sep_token_id, tf.dtypes.int32)
        self.pad_token = ref.pad_token
        self.pad_token_id = tf.constant(ref.pad_token_id, tf.dtypes.int32)
        self.cls_token = ref.cls_token
        self.cls_token_id = tf.constant(ref.cls_token_id, tf.dtypes.int32)
        self.mask_token = ref.mask_token
        self.mask_token_id = tf.constant(ref.mask_token_id, tf.dtypes.int32)

        self.max_len_single_sentence = tf.constant(ref.max_len_single_sentence, tf.dtypes.int32)
        self.max_len_sentences_pair = tf.constant(ref.max_len_sentences_pair, tf.dtypes.int32)

        self._vocabulary = [
            ref.ids_to_tokens[i] for i in range(len(ref.vocab))
        ]
        self._tokenizer = tftx.FastWordpieceTokenizer(
            self._vocabulary,
            suffix_indicator='##',
            token_out_type=tf.dtypes.int32,
            unknown_token=self.unk_token,
            support_detokenization=True)

    @cached_property
    def vocab_size(self) -> tf.Tensor:
        """Vocabulary size
        """
        return tf.constant(len(self._vocabulary), tf.dtypes.int32)

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
            'token_type_ids': self.pad_token_id,
            'attention_mask': tf.cast(self.pad_token_id, dtype=tf.dtypes.int8)
        }

    @property
    def vocab(self) -> List[str]:
        """Return vocabulary
        """
        return list(self._vocabulary)

    @tf.function
    def encode(self, text: tf.Tensor) -> Union[tf.RaggedTensor, tf.Tensor]:
        """Tokenizes and encodes a text tensor

        Args:
            text (tf.Tensor): text tensor, can be batched

        Returns:
            Union[tf.RaggedTensor, tf.Tensor]: tensor of token ids, ragged if input is batched
        """
        # Remove control characters
        # soft-hyphen: [\xad]
        # control characters: [\x00-\x1f\x7f-\x9f]
        # private use area: [\x{E000}-\x{F8FF}]
        text = tf.strings.regex_replace(
            input=text,
            pattern=r'[\x{00ad}\x{0000}-\x{001f}\x{007f}-\x{009f}\x{E000}-\x{F8FF}]',
            rewrite=r'')

        # This returns a RaggedTensor of tf.string elements
        ids = self._tokenizer.tokenize(text)

        # truncate sentence
        ids = ids[..., :self.max_len_single_sentence]

        # add [CLS] and [SEP] token
        ids = tftx.pad_along_dimension(
            ids,
            left_pad=tf.expand_dims(self.cls_token_id, -1),
            right_pad=tf.expand_dims(self.sep_token_id, -1)
        )

        return ids

    @tf.function
    def __call__(self, text: tf.Tensor) -> TokenizedDict:
        """Tokenizes and encodes text, then annotates with the type and mask

        Args:
            text (tf.Tensor): text tensor, can be batched

        Returns:ø
            TokenizedDict: Dict of input_ids, token_type_ids, and attention_mask
        """
        input_ids = self.encode(text)
        return {
            'input_ids': input_ids,
            'token_type_ids': tf.zeros_like(input_ids),
            'attention_mask': tf.ones_like(input_ids, dtype=tf.dtypes.int8)
        }

    @tf.function
    def decode(self, ids: tf.Tensor) -> tf.Tensor:
        """Decode token ids

        Args:
            ids (tf.Tensor): token ids

        Returns:
            tf.Tensor: decoded token ids as a text tensor
        """
        # Compute sequence length for each observation
        ids_with_safe_eos = tf.concat([
            ids,
            tf.fill((tf.shape(ids)[0], 1), self.sep_token_id)
        ], axis=-1)

        # Use argmax to find the first True. Note that ids_with_safe_eos
        # guarantees that there is at least one True.
        sequence_lengths = tf.argmax(
            tf.cast(tf.equal(ids_with_safe_eos, self.sep_token_id), tf.dtypes.int8),
            axis=-1)

        # Use sequence lengths to create a RaggedTensor, this will not
        # contain the [EOS] token. Since the length is acutally an index.
        ids_ragged = tf.RaggedTensor.from_tensor(ids, lengths=sequence_lengths)

        # Convert ragged token IDs to tokens and join
        return self._tokenizer.detokenize(ids_ragged)
