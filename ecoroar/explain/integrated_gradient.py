
import tensorflow as tf

from ._importance_measure import ImportanceMeasure


class IntegratedGradientSignExplainer(ImportanceMeasure):
    _name = 'int-grad-sign'
    _implements_explain_batch = True

    def __init__(self, *args, riemann_samples=20, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.riemann_samples = tf.convert_to_tensor(riemann_samples, dtype=tf.dtypes.int32)

    def _explain_batch(self, x, y):
        dtype = self._model.embedding_matrix.dtype

        # Prepear a compact embedding matrix for doing sum(x * dy/dz @ W.T) efficently.
        embedding_matrix_compact = tf.gather(self._model.embedding_matrix, x['input_ids'])  # (B, T, E)

        # Prepear a contuinues input that we can differentiate againist
        x_embed = self._model.inputs_embeds(x)

        # Prepear baseline, most papers uses the pad token embedding (usually 0). Properly not a
        # great choice. But the aim here is just to evaluate what they are doing.
        baseline = tf.reshape(self._model.embedding_matrix[self._tokenizer.pad_token_id, :], (1, 1, -1))

        # Intialize a mean for holding the average
        online_mean = tf.zeros(tf.shape(x['input_ids']),
                               dtype=dtype)
        for riemann_step in tf.range(1, self.riemann_samples + 1):
            embedding_scale = tf.cast(riemann_step, dtype) / tf.cast(self.riemann_samples, dtype)

            # Compute dy(z*s)/dz
            with tf.GradientTape(watch_accessed_variables=False) as g:
                g.watch(x_embed['inputs_embeds'])
                inputs_embeds_scaled = baseline + embedding_scale * (x_embed['inputs_embeds'] - baseline)

                logits = self._model({
                    key: inputs_embeds_scaled if key == 'inputs_embeds' else value
                    for key, value in x_embed.items()
                }).logits
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
            ) # (B, T)

            # Update the online mean (Knuth Algorithm), this is more memory
            # efficient that storing x_yc_wrt_x for each Riemann step.
            online_mean += (yc_wrt_x_compact - online_mean)/tf.cast(riemann_step, dtype)

        # Return the signed explanation
        return online_mean  # (B, T)


class IntegratedGradientAbsExplainer(IntegratedGradientSignExplainer):
    _name = 'int-grad-abs'

    def _explain_batch(self, x, y):
        yc_wrt_x_compact = super()._explain_batch(x, y)
        return tf.math.abs(yc_wrt_x_compact)  # (B, T)
