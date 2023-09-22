
import tensorflow as tf
import numpy as np

from ecoroar.model import SimpleTestModel


def test_simple_model():
    model = SimpleTestModel()
    model.compile()
    logits = model({
        'input_ids': tf.constant([
            [0, 3, 3, 1, 2],
            [0, 3, 1, 2, 2]
        ], dtype=tf.dtypes.int32),
        'attention_mask': tf.constant([
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0]
        ], dtype=tf.dtypes.int8)
    }).logits

    np.testing.assert_allclose(logits, [
        [2, 2, 4],
        [1, 2, 3]
    ])
