
from abc import ABC, abstractmethod

import tensorflow as tf
from ..types import TokenizedDict, Tokenizer, Model

class ImportanceMeasure(ABC):
    name: str
    implements_explain_batch: bool = False

    def __init__(self, tokenizer: Tokenizer, model: Model) -> None:
        super().__init__()
        self._tokenizer = tokenizer
        self._model = model

    def _explain_observation(self, x: TokenizedDict, y: tf.Tensor) -> tf.Tensor:
        """Explains a single observation.

        Args:
            x (TokenizedDict): he input to the model, each part has shape [sequence_length]
            y (tf.Tensor): The output label to explain, is a scalar

        Returns:
            tf.Tensor: Returns explanations as a flat tensor, should have shape [sequence_length].
        """
        raise NotImplementedError('_explain_observation is not implemented.'
                                  ' Either _explain_observation or _explain_batch should be implemented.')

    def _wrap_explain_observation(self, x: TokenizedDict, y: tf.Tensor) -> tf.RaggedTensor:
        """Calls _explain_observation for each observation in the batch

        Args:
            x (TokenizedDict): The input to the model, each part has shape [batch_size, sequence_length]
            y (tf.Tensor): The output label to explain, has shape [batch_size]

        Returns:
            tf.RaggedTensor: Returns explanations for each observation, as rows in a RaggedTensor.
                Note, that by default the tokenizer.padding_values were used to infer the sequence_length.
        """
        batch_size = tf.shape(y)[0]
        dtype = x['input_ids'].dtype

        # Convert input to RaggedTensor structure
        ragged = tf.nest.map_structure(
            lambda tensor, padding_value: tf.RaggedTensor.from_tensor(tensor, padding=padding_value),
            x, self._tokenizer.padding_values
        )
        sequence_lengths = ragged['input_ids'].row_lengths()

        # For each observation, call self._explain_observation() and store the results in a TensorArray
        explain_all = tf.TensorArray(dtype, size=batch_size, infer_shape=False, element_shape=(None, ))
        for obs_i in tf.range(batch_size):
            # Get observation i
            obs_x = tf.nest.map_structure(lambda ragged_tensor: ragged_tensor[obs_i, ...], ragged)
            obs_y = y[obs_i]

            # Explain observation and check that the results has the correct size before saving the data
            obs_explain = self._explain_observation(obs_x, obs_y)
            tf.debugging.assert_equal(
                sequence_lengths[obs_i], tf.shape(obs_explain)[0],
                message='explanation has correct length'
            )
            explain_all = explain_all.write(obs_i, obs_explain)

        # Convert the tensorArray to a RaggedTensor
        return tf.RaggedTensor.from_row_lengths(
            values=explain_all.concat(),
            row_lengths=sequence_lengths
        )

    def _explain_batch(x: TokenizedDict, y: tf.Tensor) -> tf.Tensor:
        """Explains the entire batch.

        Args:
            x (TokenizedDict): The input to the model, each part has shape [batch_size, sequence_length]
            y (tf.Tensor): The output label to explain, has shape [batch_size]

        Returns:
            tf.Tensor: Returns explanations for each observation.
        """

        raise NotImplementedError('_explain_batch is not implemented.'
                            ' Either _explain_observation or _explain_batch should be implemented.')

    def _wrap_explain_batch(self, x: TokenizedDict, y: tf.Tensor) -> tf.RaggedTensor:
        """Converts the tensor output of _explain_batch to a RaggedTensor.

        This uses the inputs (x) as the template for infering the sequence_length.

        Args:
            x (TokenizedDict): The input to the model, each part has shape [batch_size, sequence_length]
            y (tf.Tensor): The output label to explain, has shape [batch_size]

        Returns:
            tf.RaggedTensor: Returns explanations for each observation, as rows in a RaggedTensor.
                Note, that by default the tokenizer.padding_values were used to infer the sequence_length.
        """
        explain = self._explain_batch(x, y)
        sequence_length = tf.math.reduce_sum(
            tf.cast(x['input_ids'] != self._tokenizer.padding_values['input_ids'], dtype=tf.dtypes.int32),
            axis=1
        )
        return tf.RaggedTensor.from_tensor(explain, lengths=sequence_length)

    def __call__(self, x: TokenizedDict, y: tf.Tensor) -> tf.RaggedTensor:
        """Explains the model given the input-output pair

        Args:
            x (TokenizedDict): The input to the model, each part has shape [batch_size, sequence_length]
            y (tf.Tensor): The output label to explain, has shape [batch_size]

        Returns:
            tf.RaggedTensor: Returns explanations for each observation, as rows in a RaggedTensor.
                Note, that by default the tokenizer.padding_values were used to infer the sequence_length.
        """

        if self.implements_explain_batch:
            return self._wrap_explain_batch(x, y)
        else:
            return self._wrap_explain_observation(x, y)
