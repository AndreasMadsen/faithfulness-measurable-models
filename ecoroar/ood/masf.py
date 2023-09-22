
from typing import Tuple

import tensorflow as tf
import numpy as np

from ..types import TokenizedDict, Tokenizer, Model
from ..util import get_compiler
from ..transform import MapOnGPU


@tf.function(jit_compile=True, reduce_retracing=True)
def _emerical_cdf_scan_reducer(counts, batch, samples):
    return counts + tf.reduce_sum(
        tf.cast(tf.expand_dims(batch, 0) < tf.expand_dims(samples, 1), tf.dtypes.int32),
        axis=1)


def _emerical_cdf_scan(dist: tf.data.Dataset, samples: tf.Tensor) -> tf.Tensor:
    """Evalute the CDF of samples

    Args:
        dist (tf.data.Dataset): Distributional representation.
        samples (tf.Tensor): Samples with shape [N, ...]

    Returns:
        tf.Tensor: probablity, with shape [N, ...]
    """
    num_of_obs = tf.constant(0)
    counts = tf.zeros(tf.shape(samples), dtype=tf.int32)

    for batch in dist:
        num_of_obs = num_of_obs + tf.shape(batch)[0]
        counts = _emerical_cdf_scan_reducer(counts, batch, samples)

    # convert indices to probabilites by dividing
    probs = counts / num_of_obs
    return tf.cast(probs, dtype=tf.dtypes.float32)


def _reduce_simes(p_values: tf.Tensor, axis: int = -1) -> tf.Tensor:
    """Implements Simes reduction along the axis

    Args:
        p_values (tf.Tensor): p-values to reduce.
        axis (int, optional): Axes to reduce dimentions along. Defaults to -1.

    Returns:
        tf.Tensor: Simes test-statistics
    """
    num_of_p_values = tf.shape(p_values)[axis]
    p_values_sorted = tf.sort(p_values, axis=axis)
    ratios = tf.cast(num_of_p_values / tf.range(1, num_of_p_values + 1), dtype=tf.dtypes.float32)
    test_statistic = tf.math.reduce_min(p_values_sorted * ratios, axis=axis)
    return test_statistic


def _reduce_fisher(p_values: tf.Tensor, axis: int = -1, eps: float = 1e-8) -> tf.Tensor:
    """Implements Fisher reduction along the axis

    Args:
        p_values (tf.Tensor): p-values to reduce.
        axis (int, optional): Axes to reduce dimentions along. Defaults to -1.
        eps (float, optional): To avoid nan values, due to log(0). Instead use log(p + eps). Defaults to 1e-8.

    Returns:
        tf.Tensor: Fisher test-statistics
    """
    return -2 * tf.math.reduce_sum(tf.math.log(p_values + eps), axis=axis)


def _two_sided_p_value(cdf_values: tf.Tensor) -> tf.Tensor:
    """Converts CDF probabilies (technically also p-values) to two-sided p-values.

    Args:
        p_values (tf.Tensor): CDF probabilities to convert.

    Returns:
        tf.Tensor: two-sided p-values
    """
    return tf.math.minimum(cdf_values, 1 - cdf_values)


class MaSF():
    _name: str = 'masf'

    def __init__(self, tokenizer: Tokenizer, model: Model,
                 verbose=True,
                 batch_size: Tuple[int, int, int] = (2**10, 2**13, 2**16),
                 run_eagerly: bool = False,
                 jit_compile: bool = False) -> None:
        """Implementation of MaSF

        MaSF is an out-of-distribution detection methods, that uses non-parametic
          statistics to provide right-sided p-values for new observations.

        Paper: https://openreview.net/forum?id=Oy9WeuZD51

        Args:
            tokenizer (Tokenizer): Tokenizer used to inform about padding tokens
            model (Model): Model to infer OOD statistics on
            batch_size (Tuple[int, int, int], optional). The batch_size after each reduce step.
            run_eagerly (bool, optional): If True, tf.function is not used. Defaults to False.
            jit_compile (bool, optional): If True, XLA compiling is enabled. Defaults to False.
        """
        self._model = model
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._verbose = verbose

        self._emperical_distribution_level_1 = None
        self._emperical_distribution_level_2 = None
        self._emperical_distribution_level_3 = None

        # compile functions, this is more complex than usual, because .fit is typically only called once
        #   (and thus won't be compiled). However, it is quite expensive. So, instead it's subroutines are
        #   compiled.
        maybe_jit_compiler = get_compiler(run_eagerly, jit_compile)
        maybe_std_compiler = get_compiler(run_eagerly, False)

        self._wrap_get_hidden_state_signal = maybe_jit_compiler(self._get_hidden_state_signal)
        self._wrap_level_1_to_2 = maybe_std_compiler(self._level_1_to_2)
        self._wrap_level_2_to_3 = maybe_std_compiler(self._level_2_to_3)
        self._wrap_level_3_to_p = maybe_std_compiler(self._level_3_to_p)

        self._emerical_cdf = maybe_std_compiler(_emerical_cdf_scan)
        self._reduce_simes = maybe_jit_compiler(_reduce_simes)
        self._reduce_fisher = maybe_jit_compiler(_reduce_fisher)
        self._two_sided_p_value = maybe_jit_compiler(_two_sided_p_value)

    def _get_hidden_state_signal(self, x: TokenizedDict) -> tf.Tensor:
        batch_size, sequence_length = tf.unstack(tf.shape(x['input_ids']), num=2)

        hidden_states = self._model(x, output_hidden_states=True).hidden_states  # Tuple[tf.Tensor[B, T, D]]
        hidden_states = tf.stack(hidden_states, axis=1)  # tf.Tensor[B, L, T, D]

        mask = tf.reshape(x['input_ids'] == self._tokenizer.pad_token_id,
                          (batch_size, 1, sequence_length, 1))  # tf.Tensor[B, 1, T, 1]
        hidden_states = tf.where(mask, -tf.cast(np.inf, dtype=hidden_states.dtype), hidden_states)  # tf.Tensor[B, L, T, D]
        hidden_states = tf.math.reduce_max(hidden_states, axis=2)  # tf.Tensor[B, L, D]
        return hidden_states

    # Fit emperical distributions and get p-values, OUTPUT: [N, L, D]
    def _level_1_to_2(self, samples: tf.Tensor) -> tf.Tensor:
        level_1_prop = self._emerical_cdf(self._emperical_distribution_level_1, samples)
        level_1_p = self._two_sided_p_value(level_1_prop)
        level_2 = self._reduce_simes(level_1_p, axis=-1)
        return level_2

    def _level_2_to_3(self, samples: tf.Tensor) -> tf.Tensor:
        level_2_prop = self._emerical_cdf(self._emperical_distribution_level_2, samples)
        level_2_p = self._two_sided_p_value(level_2_prop)
        level_3 = self._reduce_fisher(level_2_p, axis=-1)
        return level_3

    def _level_3_to_p(self, samples: tf.Tensor) -> tf.Tensor:
        level_3_prop = self._emerical_cdf(self._emperical_distribution_level_3, samples)
        level_3_p = 1 - level_3_prop
        return level_3_p

    def fit(self, dataset: tf.data.Dataset):
        """Builds the distributional knoweldge to detect OOD.

        For MaSF the output label is not used.

        Args:
            dataset (tf.data.Dataset): The dataset to build distributional knoweldge from.
                Often this will be the validation dataset, using the same transforms
                as was used during training.
        """
        # Fit emperical distributions and get p-values, OUTPUT: [N, L, D]
        self._emperical_distribution_level_1 = dataset \
            .apply(MapOnGPU(
                lambda x, y: self._wrap_get_hidden_state_signal(x),
                output_signature=lambda _: tf.TensorSpec(
                    shape=[None, self._model.config.num_hidden_layers + 1, self._model.config.hidden_size],
                    dtype=tf.keras.mixed_precision.global_policy().compute_dtype,
                )
            )) \
            .rebatch(self._batch_size[0]) \
            .cache()

        # Fit emperical distributions and get p-values, OUTPUT: [N, L]
        self._emperical_distribution_level_2 = self._emperical_distribution_level_1.apply(MapOnGPU(
            self._wrap_level_1_to_2,
            output_signature=lambda _: tf.TensorSpec(
                shape=[None, self._model.config.num_hidden_layers + 1],
                dtype=tf.dtypes.float32,
            )
        )) \
            .rebatch(self._batch_size[1]) \
            .cache()

        # aggregate L-dimention of p-values using Fisher, OUTPUT: [N]
        self._emperical_distribution_level_3 = self._emperical_distribution_level_2.apply(MapOnGPU(
            self._wrap_level_2_to_3,
            output_signature=lambda _: tf.TensorSpec(
                shape=[None],
                dtype=tf.dtypes.float32,
            )
        )) \
            .rebatch(self._batch_size[2]) \
            .cache()

    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Annotates an (x, y) dataset of observations as ood

        Args:
            x (tf.data.Dataset): (x, y) dataset of observations

        Returns:
            tf.Tensor: (ood, ) dataset of of annotations
        """

        return dataset \
            .apply(MapOnGPU(
                lambda x, y: self._wrap_get_hidden_state_signal(x),
                output_signature=lambda _: tf.TensorSpec(
                    shape=[None, self._model.config.num_hidden_layers + 1, self._model.config.hidden_size],
                    dtype=tf.keras.mixed_precision.global_policy().compute_dtype,
                )
            )) \
            .rebatch(self._batch_size[0]) \
            .apply(MapOnGPU(
                self._wrap_level_1_to_2,
                output_signature=lambda _: tf.TensorSpec(
                    shape=[None, self._model.config.num_hidden_layers + 1],
                    dtype=tf.dtypes.float32,
                )
            )) \
            .rebatch(self._batch_size[1]) \
            .apply(MapOnGPU(
                self._wrap_level_2_to_3,
                output_signature=lambda _: tf.TensorSpec(
                    shape=[None],
                    dtype=tf.dtypes.float32,
                )
            )) \
            .rebatch(self._batch_size[2]) \
            .apply(MapOnGPU(
                self._wrap_level_3_to_p,
                output_signature=lambda _: tf.TensorSpec(
                    shape=[None],
                    dtype=tf.dtypes.float32,
                )
            ))
