
import tensorflow as tf

from ._importance_measure import ImportanceMeasure


class _GradientGeneralizedExplaner(ImportanceMeasure):
    _implements_explain_batch = True

    def _compute_gradient(self, x, y):
        x = self._model.inputs_embeds(x)
        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(x['inputs_embeds'])
            logits = self._model(x).logits
            logits_at_y = tf.gather(logits, y, batch_dims=1)

        # compute a batch gradient, this works because the logits_at_y are independent.
        yc_wrt_embedding = g.gradient(logits_at_y, x['inputs_embeds'])   # [B, T, E]

        # We need the gradient wrt. x. However, to compute that directly with .grad would
        # require the model input to be a one_hot encoding. Creating a one_hot encoding
        # is very memory inefficient. To avoid that, manually compute the gradient wrt. x
        # based on the gradient yc_wrt_embedding.
        # yc_wrt_x = yc_wrt_emb @ emb_wrt_x = yc_wrt_emb @ emb_matix.T
        yc_wrt_x = tf.matmul(yc_wrt_embedding, self._model.embedding_matrix, transpose_b=True)  # [B, T, V]
        return yc_wrt_x  # [B, T, V]


class GradientL2Explainer(_GradientGeneralizedExplaner):
    _name = 'grad-l2'

    def _explain_batch(self, x, y):
        yc_wrt_x = self._compute_gradient(x, y)
        return tf.norm(yc_wrt_x, ord=2, axis=2)  # [B, T]


class GradientL1Explainer(_GradientGeneralizedExplaner):
    _name = 'grad-l1'

    def _explain_batch(self, x, y):
        yc_wrt_x = self._compute_gradient(x, y)
        return tf.norm(yc_wrt_x, ord=1, axis=2)  # [B, T]
