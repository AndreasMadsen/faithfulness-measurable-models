
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from ..types import TokenizedDict, Tokenizer, Model
from ..util import get_compiler
from ..transform import MapOnGPU

def _emerical_cdf_scan(dist: tf.Tensor, samples: tf.Tensor)-> tf.Tensor:
    """Evalute the CDF of samples given the distribution from _emerical_fit_scan

    This one uses O(1) for fit and O(n) for cdf.

    Args:
        dist (tf.Tensor): Distributional representation.
        samples (tf.Tensor): Samples with shape [N, ...]

    Returns:
        tf.Tensor: probablity, with shape [N, ...]
    """
    counts = tf.vectorized_map(
        lambda sample: tf.math.reduce_sum(tf.cast(
            dist < tf.expand_dims(sample, axis=0),
            dtype=tf.dtypes.int32), axis=0),
        samples)

    # convert indices to probabilites by dividing
    num_of_obs = tf.shape(dist)[0]
    probs = counts / num_of_obs
    return tf.cast(probs, dtype=tf.dtypes.float32)

def _reduce_simes(p_values: tf.Tensor, axis: int=-1) -> tf.Tensor:
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

def _reduce_fisher(p_values: tf.Tensor, axis: int=-1, eps: float=1e-8) -> tf.Tensor:
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

class MaSFSlow():
    _name: str = 'masf-slow'

    def __init__(self, tokenizer: Tokenizer, model: Model,
                 verbose=True,
                 run_eagerly: bool = False,
                 jit_compile: bool = False) -> None:
        """Implementation of MaSF

        MaSF is an out-of-distribution detection methods, that uses non-parametic
          statistics to provide right-sided p-values for new observations.

        Paper: https://openreview.net/forum?id=Oy9WeuZD51

        Args:
            tokenizer (Tokenizer): Tokenizer used to inform about padding tokens
            model (Model): Model to infer OOD statistics on
            run_eagerly (bool, optional): If True, tf.function is not used. Defaults to False.
            jit_compile (bool, optional): If True, XLA compiling is enabled. Defaults to False.
        """
        self._model = model
        self._tokenizer = tokenizer
        self._verbose = verbose

        self._emperical_distribution_level_1 = None
        self._emperical_distribution_level_2 = None
        self._emperical_distribution_level_3 = None

        # compile functions, this is more complex than usual, because .fit is typically only called once
        #   (and thus won't be compiled). However, it is quite expensive. So, instead it's subroutines are
        #   compiled.
        compiler = get_compiler(run_eagerly, jit_compile)
        self._wrap_get_hidden_state_signal = compiler(self._get_hidden_state_signal)
        self._wrap_estimate_p_values = compiler(self._estimate_p_values)
        self._emerical_cdf = compiler(_emerical_cdf_scan)
        self._reduce_simes = compiler(_reduce_simes)
        self._reduce_fisher = compiler(_reduce_fisher)
        self._two_sided_p_value = compiler(_two_sided_p_value)

    def _get_hidden_state_signal(self, x):
        batch_size, sequence_length = tf.unstack(tf.shape(x['input_ids']), num=2)

        hidden_states = self._model(x, output_hidden_states=True).hidden_states  # Tuple[tf.Tensor[B, T, D]]
        hidden_states = tf.stack(hidden_states, axis=1)  # tf.Tensor[B, L, T, D]

        mask = tf.reshape(x['input_ids'] == self._tokenizer.pad_token_id, (batch_size, 1, sequence_length, 1)) # tf.Tensor[B, 1, T, 1]
        hidden_states = tf.where(mask, -tf.cast(np.inf, dtype=hidden_states.dtype), hidden_states)  # tf.Tensor[B, L, T, D]
        hidden_states = tf.math.reduce_max(hidden_states, axis=2)  # tf.Tensor[B, L, D]
        return hidden_states

    def fit(self, dataset: tf.data.Dataset):
        """Builds the distributional knoweldge to detect OOD.

        For MaSF the output label is not used.

        Args:
            dataset (tf.data.Dataset): The dataset to build distributional knoweldge from.
                Often this will be the validation dataset, using the same transforms
                as was used during training.
        """
        # atempt to infer size
        size = dataset.cardinality()
        if size == tf.data.INFINITE_CARDINALITY or size == tf.data.UNKNOWN_CARDINALITY:
            size == None

        # Acumulate the hidden_states
        accumulating_hidden_states = tf.TensorArray(
            tf.keras.mixed_precision.global_policy().compute_dtype,
            size=size,
            infer_shape=False,
            element_shape=(None, self._model.config.num_hidden_layers + 1, self._model.config.hidden_size)
        )
        for i, (x, _) in tqdm(dataset.enumerate(), desc='accumulating statistics', disable=not self._verbose):
            hidden_states = self._wrap_get_hidden_state_signal(x)
            accumulating_hidden_states = accumulating_hidden_states.write(i, hidden_states)

        # Fit emperical distributions and get p-values
        level_1_test_statistics = accumulating_hidden_states.concat()  # [N, L, D]
        self._emperical_distribution_level_1 = level_1_test_statistics
        level_1_prob = self._emerical_cdf(self._emperical_distribution_level_1, level_1_test_statistics)
        level_1_p = self._two_sided_p_value(level_1_prob)

        # aggregate D-dimention of p-values using Simes
        level_2_test_statistics = self._reduce_simes(level_1_p, axis=-1)  # [N, L]
        # Fit emperical distributions and get p-values
        self._emperical_distribution_level_2 = level_2_test_statistics
        level_2_prob = self._emerical_cdf(self._emperical_distribution_level_2, level_2_test_statistics)
        level_2_p = self._two_sided_p_value(level_2_prob)

        # aggregate L-dimention of p-values using Fisher
        level_3_test_statistics = self._reduce_fisher(level_2_p, axis=-1)  # [N]
        # Fit emperical distributions
        self._emperical_distribution_level_3 = level_3_test_statistics

    def _estimate_p_values(self, x: TokenizedDict) -> tf.Tensor:
        # Use max-aggregated hidden states as the inital test-statistic
        level_1_test_statistics = self._wrap_get_hidden_state_signal(x)  # [N, L, D]

        # Get p-values
        level_1_prob = self._emerical_cdf(self._emperical_distribution_level_1, level_1_test_statistics)
        level_1_p = self._two_sided_p_value(level_1_prob)

        # aggregate D-dimention of p-values using Simes and get p-values
        level_2_test_statistics = self._reduce_simes(level_1_p, axis=-1)  # [N, L]
        level_2_prob = self._emerical_cdf(self._emperical_distribution_level_2, level_2_test_statistics)
        level_2_p = self._two_sided_p_value(level_2_prob)

        # aggregate L-dimention of p-values using Fisher and get p-values
        level_3_test_statistics = self._reduce_fisher(level_2_p, axis=-1)  # [N]
        level_3_prob = self._emerical_cdf(self._emperical_distribution_level_3, level_3_test_statistics)
        # assuming right sided p-values
        # Note, it is unclear why they use right-sided p-values in the paper.
        level_3_p = 1 - level_3_prob

        return level_3_p

    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Annotates an (x, y) dataset of observations as ood

        Args:
            x (tf.data.Dataset): (x, y) dataset of observations

        Returns:
            tf.Tensor: (ood, ) dataset of of annotations
        """

        return dataset \
            .apply(MapOnGPU(
                lambda x, y: self._wrap_estimate_p_values(x),
                output_signature=lambda _: tf.TensorSpec(
                    shape=[None],
                    dtype=tf.dtypes.float32,
                )
            ))
