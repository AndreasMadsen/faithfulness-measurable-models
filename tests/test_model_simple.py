
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
        ], dtype=tf.dtypes.int32)
    })

    np.testing.assert_array_equal(logits, [
        [-2, 4, 2],
        [-1, 3, 2]
    ])
