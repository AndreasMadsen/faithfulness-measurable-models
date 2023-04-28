
import pytest
import tensorflow as tf
import numpy as np
import scipy.stats

from ecoroar.ood.masf import _emerical_cdf_scan


@pytest.mark.parametrize("shape", [(2, 2), (3, 1, 2)], ids=lambda shape: ''.join(map(str, shape)))
def test_emperical_distribution(shape):
    dist = scipy.stats.norm(1, 2)
    samples = dist.rvs((10000, *shape), random_state=0)
    batch = dist.rvs((10, *shape), random_state=0)

    emperical_dist = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(samples, dtype=tf.dtypes.float32)
    ).batch(1000)

    emperical_cdf = _emerical_cdf_scan(emperical_dist, tf.convert_to_tensor(batch, dtype=tf.dtypes.float32))
    np.testing.assert_allclose(emperical_cdf.numpy(), dist.cdf(batch), rtol=0.001, atol=0.01)
