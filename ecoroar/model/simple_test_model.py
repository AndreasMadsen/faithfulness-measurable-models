
import tensorflow as tf

from ..types import TokenizedDict

_default_embedding = [
    [0, 1],  # BOS
    [0, 1],  # EOS
    [0, 0],  # PAD
    [-1, 1]   # TEXT
]

_defalt_kernel = [
    [1, 0, 1],
    [0, 1, 1]
]

class SimpleTestModel(tf.keras.Model):
    def __init__(self, embeddings_initializer=_default_embedding, kernel_initializer=_defalt_kernel) -> None:
        super().__init__()
        self._embedding = tf.keras.layers.Embedding(
            input_dim=4,
            output_dim=2,
            embeddings_initializer=tf.keras.initializers.Constant(embeddings_initializer)
        )
        self._dense = tf.keras.layers.Dense(
            units=3,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.Constant(kernel_initializer)
        )

    def call(self, x: TokenizedDict, training=False):
        z = x['input_ids']
        z = self._embedding(z, training=training)
        z = tf.math.reduce_sum(z, axis=1)
        z = self._dense(z, training=training)
        return z
