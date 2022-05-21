
import pathlib
from dataclasses import dataclass
from typing import Callable

import pytest
import numpy as np
import tensorflow as tf
import sklearn.metrics
import scipy.special
import scipy.stats

from ecoroar.metric import AUROC, F1Score, Pearson

@dataclass
class MetricExpectPair:
    Metric: Callable[[], tf.keras.metrics.Metric]
    expect: Callable[[np.array, np.array], np.array]

metrics = [
    MetricExpectPair(
        lambda: AUROC(from_logits=True),
        lambda y_true, y_pred: sklearn.metrics.roc_auc_score(y_true[:, 0], y_pred[:, 1])
    ),
    MetricExpectPair(
        lambda: F1Score(num_classes=2, average='macro'),
        lambda y_true, y_pred: sklearn.metrics.f1_score(y_true[:, 0], np.argmax(y_pred, axis=1), average='macro')
    ),
    MetricExpectPair(
        lambda: F1Score(num_classes=2, average='micro'),
        lambda y_true, y_pred: sklearn.metrics.f1_score(y_true[:, 0], np.argmax(y_pred, axis=1), average='micro')
    ),
    MetricExpectPair(
        lambda: Pearson(),
        lambda y_true, y_pred: scipy.stats.pearsonr(y_true[:, 0], scipy.special.expit(y_pred[:, 1]))[0]
    )
]

@pytest.mark.parametrize("info", metrics, ids=lambda info: info.Metric().name)
def test_metric(info):
    metric = info.Metric()

    rng  = tf.random.Generator.from_seed(1)
    y_pred = rng.uniform([20, 2], minval=-2, maxval=2)
    y_true = rng.uniform([20, 1], minval=0, maxval=2, dtype=tf.dtypes.int64)

    # test that metric.update_state() initializes the metric correctly
    metric.update_state(y_true[:10, :], y_pred[:10, :])
    np.testing.assert_allclose(
        metric.result().numpy(),
        info.expect(y_true[:10, :].numpy(), y_pred[:10, :].numpy()),
        rtol=1e-2
    )

    # test that metric.update_state() updates the metric correctly
    metric.update_state(y_true[10:, :], y_pred[10:, :])
    np.testing.assert_allclose(
        metric.result().numpy(),
        info.expect(y_true.numpy(), y_pred.numpy()),
        rtol=1e-2
    )

    # test that metric.reset_state() resets the metric correctly
    metric.reset_state()
    metric.update_state(y_true, y_pred)
    np.testing.assert_allclose(
        metric.result().numpy(),
        info.expect(y_true.numpy(), y_pred.numpy()),
        rtol=1e-2
    )
