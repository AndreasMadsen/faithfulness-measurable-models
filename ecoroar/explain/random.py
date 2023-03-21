
import tensorflow as tf

from ._importance_measure import ImportanceMeasure


class RandomExplainer(ImportanceMeasure):
    _name = 'rand'
    _implements_explain_batch = True

    def _explain_batch(self, x, y):
        return self._rng.uniform(tf.shape(x['input_ids']))
