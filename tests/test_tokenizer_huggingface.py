
import pathlib

import pytest
import numpy as np
import tensorflow as tf
from transformers import RobertaTokenizerFast

from ecoroar.dataset import CoLADataset, RTEDataset
from ecoroar.tokenizer import HuggingfaceTokenizer


@pytest.fixture
def tokenizer_tf():
    return HuggingfaceTokenizer('roberta-base', persistent_dir=pathlib.Path('.'))


@pytest.fixture
def tokenizer_ref():
    return RobertaTokenizerFast.from_pretrained("roberta-base", cache_dir='./cache/tokenizer')


def test_padding_values(tokenizer_tf):
    np.testing.assert_array_equal(tokenizer_tf.padding_values['input_ids'].numpy(),
                                  np.array(tokenizer_tf.pad_token_id, np.int32))
    np.testing.assert_array_equal(tokenizer_tf.padding_values['attention_mask'].numpy(),
                                  np.array(0, np.int8))


def test_tokenizer_consistency(tokenizer_tf, tokenizer_ref):
    dataset = CoLADataset(persistent_dir=pathlib.Path('.'), use_cache=False, use_snapshot=False)

    for split in [dataset.train(), dataset.valid(), dataset.test()]:
        for x, y in split.take(2):
            out_tf = tokenizer_tf(x)
            out_ref = tokenizer_ref(
                x[0].numpy().decode('utf-8'),
                truncation=True, return_tensors='np'
            )

            np.testing.assert_array_equal(out_tf['input_ids'].numpy(),
                                          out_ref['input_ids'][0].astype(np.int32))

            np.testing.assert_array_equal(out_tf['attention_mask'].numpy(),
                                          out_ref['attention_mask'][0].astype(np.int8))


def test_tokenizer_paired_sequence(tokenizer_tf, tokenizer_ref):
    dataset = RTEDataset(persistent_dir=pathlib.Path('.'), use_cache=False, use_snapshot=False)

    for split in [dataset.train(), dataset.valid(), dataset.test()]:
        for x, y in split.take(2):
            out_tf = tokenizer_tf(x)
            out_ref = tokenizer_ref(
                x[0].numpy().decode('utf-8'), x[1].numpy().decode('utf-8'),
                truncation=True, return_tensors='np'
            )

            np.testing.assert_array_equal(out_tf['input_ids'].numpy(),
                                          out_ref['input_ids'][0].astype(np.int32))

            np.testing.assert_array_equal(out_tf['attention_mask'].numpy(),
                                          out_ref['attention_mask'][0].astype(np.int8))


def test_tokenizer_cardinality_kept(tokenizer_tf):
    dataset = CoLADataset(persistent_dir=pathlib.Path('.'), use_cache=False, use_snapshot=False)

    dataset_train = dataset.train() \
        .map(lambda x, y: (tokenizer_tf(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    assert dataset_train.cardinality().numpy() == 6841
