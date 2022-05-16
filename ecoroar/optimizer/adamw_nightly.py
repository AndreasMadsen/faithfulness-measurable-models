
from typing import List, Union

import tensorflow as tf


class AdamW(tf.keras.optimizers.experimental.AdamW):
    def __init__(self, learning_rate: Union[float, tf.keras.optimizers.schedules.LearningRateSchedule],
                 weight_decay: float = 0.01, epsilon: float = 1e-8,
                 exclude_from_weight_decay: List[str] = ["LayerNorm", "layer_norm", "bias"], **kwargs):
        """Wrapper on AdamWeightDecay, sets parameter exclusion and learning-rate scheduler

        * excludes bias and layer normalization parameters from the weight_decay
        * creates a linear schedule to scale the learning_rate

        Args:
            learning_rate (Union[float, tf.keras.optimizers.schedules.LearningRateSchedule]): Learning rate
            weight_decay (float, optional): Weight decay applied. Defaults to 0.01.
            epsilon (float, optional): Adam epsilon. Defaults to 1e-8.
            exclude_from_weight_decay (List[str], optional): Parameters to exclude weight decay from.
                Defaults to ["LayerNorm", "layer_norm", "bias"].
        """
        self._exclude_from_weight_decay_pattern = exclude_from_weight_decay

        super().__init__(learning_rate=learning_rate,
                         weight_decay_rate=weight_decay,
                         epsilon=epsilon,
                         **kwargs)

    def build(self, var_list):
        for var in var_list:
            print(var.name)

        return super().build(var_list)
