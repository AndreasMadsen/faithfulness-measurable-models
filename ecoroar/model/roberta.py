
import tensorflow as tf

from transformers import TFRobertaForSequenceClassification, TFRobertaPreLayerNormForSequenceClassification
from ..types import TokenizedDict, EmbeddingDict

class TFRoBERTaLikeEmbeddingAbstraction():
    def inputs_embeds(self, x: TokenizedDict, training=False) -> EmbeddingDict:
        input_ids = x['input_ids']

        # Note: tf.gather, on which the embedding layer is based, won't check positive out of bound
        # indices on GPU, returning zeros instead. This is a dangerous silent behavior.
        tf.debugging.assert_less(
            input_ids,
            tf.cast(self.config.vocab_size, dtype=input_ids.dtype),
            message=(
                "input_ids must be smaller than the embedding layer's input dimension (got"
                f" {tf.math.reduce_max(input_ids)} >= {self.config.vocab_size})"
            ),
        )
        inputs_embeds = tf.gather(params=self.embedding_matrix, indices=input_ids)

        return {
            'attention_mask': x['attention_mask'],
            'inputs_embeds': inputs_embeds
        }

class TFRoBERTaForSequenceClassificationExtra(TFRoBERTaLikeEmbeddingAbstraction, TFRobertaForSequenceClassification):
    # Argument TFRobertaForSequenceClassification with methods for managing the embedding,
    #  these methods are neccesary for computing gradients wrt. input.

    @property
    def embedding_matrix(self) -> tf.Variable:
        return self.roberta.embeddings.weight

class TFRoBERTaPreLayerNormForSequenceClassificationExtra(TFRoBERTaLikeEmbeddingAbstraction, TFRobertaPreLayerNormForSequenceClassification):
    # Argument TFRobertaPreLayerNormForSequenceClassification with methods for managing the embedding,
    #  these methods are neccesary for computing gradients wrt. input.

    @property
    def embedding_matrix(self) -> tf.Variable:
        return self.roberta_prelayernorm.embeddings.weight
