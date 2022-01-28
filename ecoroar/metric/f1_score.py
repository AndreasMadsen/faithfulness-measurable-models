
import tensorflow as tf
import tensorflow_addons as tfa

class F1Score(tfa.metrics.F1Score):
    """Make tfa.metrics.F1Score compatiable with sparse labels

    tfa.metrics.F1Score assumes the label shape to be [None, num_classes].
        However, the tasks in this project have only one label and therefore uses
        sparse labels indices of shape [None, 1]. This wrapper converts the sparse
        labels to the expected one_hot encoding.
    """

    #@tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.ensure_shape(y_true, [None, 1])
        y_true = tf.one_hot(y_true[:, 0], self.num_classes)
        super().update_state(y_true, y_pred, sample_weight=sample_weight)
