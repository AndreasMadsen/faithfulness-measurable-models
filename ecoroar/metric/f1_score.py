
import tensorflow as tf
from ._addon_f1_score import F1Score as tfa_F1Score


class F1Score(tfa_F1Score):
    """Make tfa.metrics.F1Score compatible with sparse labels

    tfa.metrics.F1Score assumes the label shape to be [None, num_classes].
        However, the tasks in this project have only one label and therefore use
        sparse labels indices of shape [None, 1]. This wrapper converts the sparse
        labels to the expected one_hot encoding.
    """

    def __init__(self, num_classes: int, average: str = 'macro', name: str = None):
        """Computes the F1-score, enther macro or micro

        Args:
            num_classes (int): Number of unique classes in the dataset.
            average (str, optional): Type of averaging to be performed on data.
                Acceptable values are `None`, `micro`, `macro`
                and `weighted`. Defaults to 'macro'.
            name (str, optional): String name of the metric instance. Default to f'{average}_f1'
        """
        if name is None:
            name = f'{average}_f1'
        super().__init__(num_classes, average=average, name=name)

    @tf.function
    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor = None):
        """Accumulates statistics for the F1-score metric.

        Args:
            y_true (tf.Tensor): The ground truth values.
            y_pred (tf.Tensor): The predicted values.
            sample_weight (tf.Tensor, optional): Optional weighting of each example. Defaults to 1.
                Can be a `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
                be broadcastable to `y_true`.
        """
        y_true = tf.ensure_shape(y_true, [None, 1])
        y_pred = tf.ensure_shape(y_pred, [None, self.num_classes])
        y_true = tf.one_hot(y_true[:, 0], self.num_classes)
        super().update_state(y_true, y_pred, sample_weight=sample_weight)
