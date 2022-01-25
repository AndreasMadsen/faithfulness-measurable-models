
import pytest
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

def test_tokenizer_consistency_single():
    tokenizer_ref = BertTokenizerFast.from_pretrained("bert-base-cased", cache_dir=f'./cache/tokenizer')
    tokenizer_tf = BertTokenizer('bert-base-cased', persistent_dir='.')
    dataset = IMDBDataset(persistent_dir='.')

    for split in [dataset.train, dataset.valid, dataset.test]:
        for example in split.take(10):
            out_tf = tokenizer_tf(example['text'])
            out_ref = tokenizer_ref(
                example['text'].numpy().decode('utf-8'),
                truncation=True, return_tensors='np'
            )

            np.testing.assert_array_equal(out_tf['input_ids'].numpy(),
                                          out_ref['input_ids'][0].astype(np.int32))
            np.testing.assert_array_equal(out_tf['token_type_ids'].numpy(),
                                          out_ref['token_type_ids'][0].astype(np.int8))
            np.testing.assert_array_equal(out_tf['attention_mask'].numpy(),
                                          out_ref['attention_mask'][0].astype(np.int8))

def test_tokenizer_consistency_batch():
    tokenizer_ref = BertTokenizerFast.from_pretrained("bert-base-cased", cache_dir=f'./cache/tokenizer')
    tokenizer_tf = BertTokenizer('bert-base-cased', persistent_dir='.')
    dataset = IMDBDataset(persistent_dir='.')

    for split in [dataset.train, dataset.valid, dataset.test]:
        for example in split.batch(1).take(10):
            out_tf = tokenizer_tf(example['text'])
            out_ref = tokenizer_ref(
                map(lambda x: x.decode('utf-8'), example['text'].numpy().tolist()),
                truncation=True, return_tensors='np'
            )

            np.testing.assert_array_equal(out_tf['input_ids'].numpy(),
                                          out_ref['input_ids'].astype(np.int32))
            np.testing.assert_array_equal(out_tf['token_type_ids'].numpy(),
                                          out_ref['token_type_ids'].astype(np.int8))
            np.testing.assert_array_equal(out_tf['attention_mask'].numpy(),
                                          out_ref['attention_mask'].astype(np.int8))

def test_tokenizer_cardinality_kept():
    tokenizer = BertTokenizer('bert-base-cased', persistent_dir='.')
    dataset = IMDBDataset(persistent_dir='.')

    dataset_train = dataset.train \
        .map(lambda item: (tokenizer(item['text']), item['label']),
             num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

    assert tf.data.experimental.cardinality(dataset_train).numpy() == 20000
