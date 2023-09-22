
import tensorflow as tf

from ._importance_measure import ImportanceMeasureBatch


class InputTimesGradientSignExplainer(ImportanceMeasureBatch):
    _name = 'inp-grad-sign'
    _signed = True
    _base_name = 'inp-grad'

    def _explain_batch(self, x, y):
        # Prepear a compact embedding matrix for doing sum(x * dy/dz @ W.T) efficently.
        embedding_matrix_compact = tf.gather(self._model.embedding_matrix, x['input_ids'])  # (B, T, E)

        x_embed = self._model.inputs_embeds(x)
        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(x_embed['inputs_embeds'])
            logits = self._model(x_embed).logits
            logits_at_y = tf.gather(logits, y, batch_dims=1)

        # compute a batch gradient, this works because the logits_at_y are independent.
        yc_wrt_embedding = g.gradient(logits_at_y, x_embed['inputs_embeds'])  # (B, T, E)

        # We need the gradient wrt. x. However, to compute that directly with .grad would
        # require the model input to be a one_hot encoding. Creating a one_hot encoding
        # is very memory inefficient. To avoid that, manually compute the gradient wrt. x
        # based on the gradient yc_wrt_embedding. That is: dy/dx = dy/dz @ W.T
        #
        # However, this require storing (B, T, V) elements which is quite a lot.
        # Since we are anyway computing a sum over the vocabulary, ie. sum(one_hot(x) * dy/dz @ W.T)
        # there is no need to store all this. Futhermore, because because x is one_hot,
        # hence there is no need to compute all the dy/dx = dy/dz @ W.T elements, where x = 0,
        # because they will anyway go away after sum.
        #
        # This einsum computes a batched inner-product
        #  (..., Z)        (..., V)                                                        (...,)
        # [z1, z2, z3] -> [z1*e11 + z2*e12 + z3*e13, -> [1*(z1*e11 + z2*e12 + z3*e13), -> [z1*e11 +
        #                  z2*e21 + z2*e22 + z3*e23,     0*(z2*e21 + z2*e22 + z3*e23),     z2*e12 +
        #                  z2*e31 + z2*e32 + z3*e33,     0*(z2*e31 + z2*e32 + z3*e33),     z3*e13]
        #                  z2*e41 + z2*e42 + z3*e43]     0*(z2*e41 + z2*e42 + z3*e43)]
        yc_wrt_x_compact = tf.linalg.einsum(
            '...i,...i->...', yc_wrt_embedding, embedding_matrix_compact,
            optimize='optimal'
        )  # (B, T)

        # Return the signed explanation
        return yc_wrt_x_compact


class InputTimesGradientAbsExplainer(InputTimesGradientSignExplainer):
    _name = 'inp-grad-abs'
    _signed = False

    def _explain_batch(self, x, y):
        yc_wrt_x_compact = super()._explain_batch(x, y)
        return tf.math.abs(yc_wrt_x_compact)  # (B, T)
