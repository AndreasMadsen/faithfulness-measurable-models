
import tensorflow as tf


class ExtraValidationDataset(tf.keras.callbacks.Callback):
    def __init__(self, *args, name='extra_val', verbose=None, **kwargs):
        """Enables to run extra validation datasets during training

        These extra validation datasets will be added to the history object.
        However, they can not be used for early stopping. Make the early
        stopping dataset the primary validation dataset.

        The arguments to the class initalizer are forwarded to `self.model.evaluate()`,
        with the exception of `name` which is used to prefix the results in the history object.

        Args:
            name (str, optional): The name of the validation dataset. Defaults to 'extra_val'.
            verbose (_type_, optional): Enables verbose mode in `model.evaluate()`. Defaults to None.
        """
        super().__init__()
        self._name = name
        self._verbose = verbose
        self._evaluate_args = args
        self._evaluate_kwargs = kwargs

    def on_epoch_end(self, epoch, logs=None):
        results = self.model.evaluate(
            *self._evaluate_args,
            return_dict=True,
            verbose=self.params['verbose'] if self._verbose is None else self._verbose,
            **self._evaluate_kwargs
        )

        for metric_name, result in results.items():
            logs[f'{self._name}_{metric_name}'] = result
