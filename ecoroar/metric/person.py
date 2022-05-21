import tensorflow as tf


class Covariance(tf.keras.metrics.Metric):
    """Metric class, computing the unbiased coveriance of two scalars.

    Algorithm from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online
    as used in:
        https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/contrib/metrics/python/ops/metric_ops.py#L3276
        https://github.com/allenai/allennlp/blob/67f32d3f1c2a4eb1301f6d858c89a7df9270e8a4/allennlp/training/metrics/covariance.py#L16

    The algorithm is:
        C_AB = C_A + C_B + (E[pred_A] - E[pred_B]) * (E[true_A] - E[true_B]) * n_A * n_B / n_AB
    where A indicates the current estimate, and B indicates the added estimate.
    """
    def __init__(self):
        """Metric for computing the covariance

        An online algorithm will be used.
        """
        self._pred_mean = tf.keras.metrics.Mean(name='prediction_mean')
        self._true_mean = tf.keras.metrics.Mean(name='label_mean')
        self._co_moment = self.add_weight(name='co_moment', shape=[], initializer='zeros')
        self._count = self.add_weight(name='count', shape=[], initializer='zeros')

    @tf.function
    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor=None):
        """Update the internal first and second order momentums

        Args:
            y_true (tf.Tensor): The ground truth values, shape=(batch_size, ).
            y_pred (tf.Tensor): The predicted values, shape=(batch_size, ).
            sample_weight (tf.Tensor, optional): Not supported.
        """
        assert sample_weight is None
        y_true = tf.ensure_shape(y_true, [None])
        y_pred = tf.ensure_shape(y_pred, [None])

        add_count = tf.size(y_true)
        previous_count = self._count
        updated_count = self._count.assign_add(add_count)

        previous_pred_mean = self._pred_mean.result()
        updated_pred_mean = self._pred_mean.update_state(y_pred)
        add_pred_mean = updated_pred_mean - previous_pred_mean

        previous_true_mean = self._true_mean.result()
        updated_true_mean = self._true_mean.update_state(y_true)
        add_true_mean = updated_true_mean - previous_true_mean

        add_co_moment = tf.math.reduce_sum((y_true - previous_true_mean) * (y_pred - updated_pred_mean))
        self._co_moment.assign_add(add_co_moment)

    @tf.function
    def result(self) -> tf.Tensor:
        """Computes unbiased covariance

        Returns:
            tf.Tensor: covariance
        """
        # All version of the online covariance indicates that it is not possible
        # to do this part of the computation in an online manner. I suppose
        # that does lead to some overflow concerns.
        return self._co_moment / (self._count - 1)


class PearsonCorrelation(Metric):
    def __init__(self) -> None:
        """Metric for computing the pearson correlation

        An online algorithm will be used.
        """
        self._covariance = Covariance()
        self._pred_variance = Covariance()
        self._true_variance = Covariance()

    @tf.function
    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor=None):
        """Updates internal covariance and variances

        Args:
            y_true (tf.Tensor): The ground truth values, shape=(batch_size, 1).
            y_pred (tf.Tensor): The predicted values, shape=(batch_size, 2).
            sample_weight (tf.Tensor, optional): Not supported.
        """
        y_true = tf.ensure_shape(y_true, [None, 1])[:, 0]
        y_pred = tf.ensure_shape(y_pred, [None, 2])[:, 1]

        self._covariance.update_state(y_true, y_pred)
        self._pred_variance.update_state(y_pred, y_pred)
        self._true_variance.update_state(y_true, y_true)

    @tf.function
    def result(self) -> tf.Tensor:
        """Computes the unbiased Pearson correlation coefficient

        Returns:
            tf.tensor: Pearson correlation coefficient
        """
        covariance = self._covariance.result()
        pred_variance = self._pred_variance.result()
        true_variance = self._true_variance.result()

        return covariance / (tf.math.sqrt(pred_variance) * tf.math.sqrt(true_variance))
