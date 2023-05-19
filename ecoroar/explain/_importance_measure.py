
from abc import ABC, abstractmethod

import tensorflow as tf

from ..types import TokenizedDict, Tokenizer, Model
from ..util import get_compiler


class ImportanceMeasure(ABC):
    _name: str
    _defer_jit: bool = False

    def __init__(self, tokenizer: Tokenizer, model: Model,
                 seed: int = None,
                 inference_batch_size: int = 16,
                 run_eagerly: bool = False,
                 jit_compile: bool = False) -> None:
        """Computes explanation where each input token is given an attribution score.

        Args:
            tokenizer (Tokenizer): The tokenizer used, this is for defining padding.
            model (Model): The model used, explanations are produced by probing the model.
            seed (int): Seed uses for some explanation methods, for example random explanation. Defaults to None.
            inference_batch_size (int, optional). The internal batch size to use at inference time.
                This may be higher than the input batch size for improved performance, in some cases.
            run_eagerly (bool, optional): If True, tf.function is not used. Defaults to False.
            jit_compile (bool, optional): If True, XLA compiling is enabled. Defaults to False.

        Raises:
            ValueError: when both run_eagerly and jit_compile are True
        """
        super().__init__()
        self._tokenizer = tokenizer
        self._model = model
        self._inference_batch_size = tf.convert_to_tensor(inference_batch_size, dtype=tf.dtypes.int32)

        if seed is None:
            self._rng = tf.random.Generator.from_non_deterministic_state()
        else:
            self._rng = tf.random.Generator.from_seed(seed)

        # setup explainer function
        std_compiler = get_compiler(run_eagerly, False)
        jit_compiler = get_compiler(run_eagerly, jit_compile and not self._defer_jit)

        if hasattr(self, '_explain_observation'):
            self._wrap_explain_observation = jit_compiler(self._explain_observation)
        if hasattr(self, '_explain_batch'):
            self._wrap_explain_batch = jit_compiler(self._explain_batch)
        self._wrap_explain = std_compiler(self._explain)

    @property
    def name(self) -> str:
        """Name of the dataset
        """
        return self._name

    def __call__(self, x: TokenizedDict, y: tf.Tensor) -> tf.RaggedTensor:
        """Explains the model given the input-output pair

        Args:
            x (TokenizedDict): The input to the model, each part has shape [batch_size, sequence_length]
            y (tf.Tensor): The output label to explain, has shape [batch_size]

        Returns:
            tf.RaggedTensor: Returns explanations for each observation, as rows in a RaggedTensor.
                Note, that by default the tokenizer.padding_values were used to infer the sequence_length.
        """

        return self._wrap_explain(x, y)


class ImportanceMeasureObservation(ImportanceMeasure):
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

    def _explain(self, x: TokenizedDict, y: tf.Tensor) -> tf.RaggedTensor:
        """Calls _explain_observation for each observation in the batch

        Args:
            x (TokenizedDict): The input to the model, each part has shape [batch_size, sequence_length]
            y (tf.Tensor): The output label to explain, has shape [batch_size]

        Returns:
            tf.RaggedTensor: Returns explanations for each observation, as rows in a RaggedTensor.
                Note, that by default the tokenizer.padding_values were used to infer the sequence_length.
        """
        batch_size = tf.shape(y)[0]

        # Convert input to RaggedTensor structure
        ragged = tf.nest.map_structure(
            lambda tensor, padding_value: tf.RaggedTensor.from_tensor(tensor, padding=padding_value),
            x, self._tokenizer.padding_values
        )
        sequence_lengths = ragged['input_ids'].row_lengths()

        # NOTE: Consider dropping this loop, it may be possible to do inference-only even for 512 token
        #   long inputs. Could be tested with MIMIC-d.
        # For each observation, call self._explain_observation() and store the results in a TensorArray
        explain_all = tf.TensorArray(tf.dtypes.float32, size=batch_size, infer_shape=False, element_shape=(None, ))
        for obs_i in tf.range(batch_size):
            # Get observation i
            obs_x = tf.nest.map_structure(lambda ragged_tensor: ragged_tensor[obs_i, ...], ragged)
            obs_y = y[obs_i]

            # Explain observation and check that the results has the correct size before saving the data
            obs_explain = self._wrap_explain_observation(obs_x, obs_y)
            tf.debugging.assert_equal(
                sequence_lengths[obs_i], tf.shape(obs_explain, out_type=tf.dtypes.int64)[0],
                message='explanation has correct length'
            )
            explain_all = explain_all.write(obs_i, tf.cast(obs_explain, dtype=tf.dtypes.float32))

        # Convert the tensorArray to a RaggedTensor
        return tf.RaggedTensor.from_row_lengths(
            values=explain_all.concat(),
            row_lengths=sequence_lengths
        )


class ImportanceMeasureBatch(ImportanceMeasure):
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

    def _explain(self, x: TokenizedDict, y: tf.Tensor) -> tf.RaggedTensor:
        """Converts the tensor output of _explain_batch to a RaggedTensor.

        This uses the inputs (x) as the template for infering the sequence_length.

        Args:
            x (TokenizedDict): The input to the model, each part has shape [batch_size, sequence_length]
            y (tf.Tensor): The output label to explain, has shape [batch_size]

        Returns:
            tf.RaggedTensor: Returns explanations for each observation, as rows in a RaggedTensor.
                Note, that by default the tokenizer.padding_values were used to infer the sequence_length.
        """
        explain = tf.cast(self._wrap_explain_batch(x, y), dtype=tf.dtypes.float32)
        sequence_length = tf.math.reduce_sum(
            tf.cast(x['input_ids'] != self._tokenizer.pad_token_id, dtype=tf.dtypes.int32),
            axis=1
        )
        return tf.RaggedTensor.from_tensor(explain, lengths=sequence_length)
