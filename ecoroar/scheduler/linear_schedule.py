
import tensorflow as tf


class LinearSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, learning_rate: float, num_training_steps: int, warm_up_until: float = 0.06):
        """Creates a warm-up-warm-down learning rate scheduler

        Args:
            learning_rate (float): max learning rate
            num_training_steps (int): total number of steps (i.e. epochs*mini_batches)
            warm_up_until (float, optional): For how many steps (relative) to warm up. Defaults to 6%.
        """
        self.learning_rate = tf.constant(learning_rate, dtype=tf.dtypes.float32)
        self.num_training_steps = tf.constant(num_training_steps, dtype=tf.dtypes.float32)
        self.num_warmup_steps = tf.constant(int(num_training_steps * warm_up_until), dtype=tf.dtypes.float32)
        self.num_warmdown_steps = tf.math.maximum(1, self.num_training_steps - self.num_warmup_steps)

    def _lr_lambda(self, current_step: tf.Tensor):
        current_step = tf.cast(current_step, tf.dtypes.float32)

        if self.num_warmup_steps == 0:
            return current_step
        if current_step < self.num_warmup_steps:
            return current_step / self.num_warmup_steps
        return tf.math.maximum(
            0.0, (self.num_training_steps - current_step) / self.num_warmdown_steps
        )

    @tf.function
    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        """Computes learning-rate

        Args:
            step (tf.Tensor): The current training step out of num_training_steps

        Returns:
            tf.Tensor: learning-rate
        """
        return self.learning_rate * self._lr_lambda(step)
