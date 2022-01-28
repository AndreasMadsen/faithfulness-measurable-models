
import numpy as np
import tensorflow as tf

from transformers import BertTokenizerFast

from ecoroar.dataset import IMDBDataset
from ecoroar.tokenizer import BertTokenizer


def test_padding_values():
    tokenizer = BertTokenizer('bert-base-cased', persistent_dir='.')
    np.testing.assert_array_equal(tokenizer.padding_values['input_ids'].numpy(),
                                  np.array(0, np.int32))
    np.testing.assert_array_equal(tokenizer.padding_values['token_type_ids'].numpy(),
                                  np.array(0, np.int32))
    np.testing.assert_array_equal(tokenizer.padding_values['attention_mask'].numpy(),
                                  np.array(0, np.int8))


def test_tokenizer_consistency():
    tokenizer_ref = BertTokenizerFast.from_pretrained("bert-base-cased", cache_dir='./cache/tokenizer')
    tokenizer_tf = BertTokenizer('bert-base-cased', persistent_dir='.')
    dataset = IMDBDataset(persistent_dir='.')

    for split in [dataset.train, dataset.valid, dataset.test]:
        for example in split.take(2):
            out_tf_single = tokenizer_tf(example['text'])
            out_tf_batch = tokenizer_tf(tf.expand_dims(example['text'], 0))
            out_ref = tokenizer_ref(
                example['text'].numpy().decode('utf-8'),
                truncation=True, return_tensors='np'
            )

            np.testing.assert_array_equal(out_tf_single['input_ids'].numpy(),
                                          out_ref['input_ids'][0].astype(np.int32))
            np.testing.assert_array_equal(out_tf_batch['input_ids'][0, :].numpy(),
                                          out_ref['input_ids'][0].astype(np.int32))

            np.testing.assert_array_equal(out_tf_single['token_type_ids'].numpy(),
                                          out_ref['token_type_ids'][0].astype(np.int8))
            np.testing.assert_array_equal(out_tf_batch['token_type_ids'][0, :].numpy(),
                                          out_ref['token_type_ids'][0].astype(np.int8))

            np.testing.assert_array_equal(out_tf_single['attention_mask'].numpy(),
                                          out_ref['attention_mask'][0].astype(np.int8))
            np.testing.assert_array_equal(out_tf_batch['attention_mask'][0, :].numpy(),
                                          out_ref['attention_mask'][0].astype(np.int8))


def test_tokenizer_cardinality_kept():
    tokenizer = BertTokenizer('bert-base-cased', persistent_dir='.')
    dataset = IMDBDataset(persistent_dir='.')

    dataset_train = dataset.train \
        .map(lambda item: (tokenizer(item['text']), item['label']),
             num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

    assert tf.data.experimental.cardinality(dataset_train).numpy() == 20000
