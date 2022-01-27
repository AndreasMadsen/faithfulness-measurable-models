
import tensorflow as tf

class AUROC(tf.keras.metrics.AUC):
    """Make keras.metrics.AUC compatiable for two-class outputs

    keras.metrics.AUC() assumes the prediction shape to be [None].
        However, most models outputs a redudant dimention for two-class problems,
        meaning the prediction shape is [None, 2]. This removes the reduandant
        class information.
    """

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.ensure_shape(y_pred, [None, 2])
        y_pred = y_pred[:, 1]
        super().update_state(y_true, y_pred, sample_weight=sample_weight)
