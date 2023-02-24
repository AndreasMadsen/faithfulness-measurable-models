
from abc import ABCMeta, abstractmethod

import tensorflow as tf

from ..types import TokenizedDict, EmbeddingDict

class Model(tf.keras.Model, metaclass=ABCMeta):
    @abstractmethod
    def inputs_embeds(self, x: TokenizedDict, training=False) -> EmbeddingDict:
        """Converts the input_ids format an inputs_embeds format

        This is useful for when the embeddings are needed before the main forward pass.

        Args:
            x (TokenizedDict): Inputs using an inputs_ids foramt
            training (bool, optional): Standard keras argument to the embedding layer. Defaults to False.

        Returns:
            EmbeddingDict: Inputs but with input_ids converted to inputs_embeds
        """
        pass

    @property
    @abstractmethod
    def embedding_matrix(self) -> tf.Variable:
        pass
