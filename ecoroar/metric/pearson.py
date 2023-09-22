import tensorflow as tf


class Covariance(tf.keras.metrics.Metric):
    """Metric class, computing the unbiased coveriance of two scalars.

    Algorithm from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online

    unlike:
        https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/contrib/metrics/python/ops/metric_ops.py#L3276
        https://github.com/allenai/allennlp/blob/67f32d3f1c2a4eb1301f6d858c89a7df9270e8a4/allennlp/training/metrics/covariance.py#L16
    this uses the weighted_batched_version. Although, weighted covariance is currently not supposed.
    """

    def __init__(self, name='pearson'):
        """Metric for computing the covariance

        An online algorithm will be used.
        """
        super().__init__(name=name)

        self._pred_mean = tf.keras.metrics.Mean(name='prediction_mean')
        self._true_mean = tf.keras.metrics.Mean(name='label_mean')
        self._co_moment = self.add_weight(name='co_moment', shape=[], initializer='zeros')
        self._count = self.add_weight(name='count', shape=[], initializer='zeros')

    @tf.function
    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor = None):
        """Update the internal first and second order momentums

        Args:
            y_true (tf.Tensor): The ground truth values, shape=(batch_size, ).
            y_pred (tf.Tensor): The predicted values, shape=(batch_size, ).
            sample_weight (tf.Tensor, optional): Not supported.
        """
        assert sample_weight is None
        y_true = tf.ensure_shape(y_true, [None])
        y_pred = tf.ensure_shape(y_pred, [None])

        self._count.assign_add(tf.size(y_true, out_type=self._count.dtype))

        self._pred_mean.update_state(y_pred)
        new_pred_mean = self._pred_mean.result()

        pre_true_mean = self._true_mean.result()
        self._true_mean.update_state(y_true)

        add_co_moment = tf.math.reduce_sum((y_true - pre_true_mean) * (y_pred - new_pred_mean))
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


class Pearson(tf.keras.metrics.Metric):
    def __init__(self, from_logits: bool = True, name='pearson') -> None:
        """Metric for computing the pearson correlation

        An online algorithm will be used.

        Args:
            from_logits (bool, optional): If y_pred is logits. Defaults to True.
        """
        super().__init__(name=name)

        self._from_logits = from_logits
        self._covariance = Covariance()
        self._pred_variance = Covariance()
        self._true_variance = Covariance()

    @tf.function
    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor = None):
        """Updates internal covariance and variances

        Args:
            y_true (tf.Tensor): The ground truth values, shape=(batch_size, 1).
            y_pred (tf.Tensor): The predicted values, shape=(batch_size, 2).
            sample_weight (tf.Tensor, optional): Not supported.
        """
        y_true = tf.ensure_shape(y_true, [None, 1])[:, 0]
        y_pred = tf.ensure_shape(y_pred, [None, 2])[:, 1]

        # Match dtypes
        y_true = tf.cast(y_true, y_pred.dtype)

        # Convert logits to probability
        if self._from_logits:
            y_pred = tf.math.sigmoid(y_pred)

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
