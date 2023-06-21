
import tensorflow as tf

from ._importance_measure import ImportanceMeasureBatch


class RandomExplainer(ImportanceMeasureBatch):
    _name = 'rand'
    _signed = False
    _base_name = 'rand'

    def _explain_batch(self, x, y):
        return self._rng.uniform(tf.shape(x['input_ids']))
