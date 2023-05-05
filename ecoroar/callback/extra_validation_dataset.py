
import tensorflow as tf

class ExtraValidationDataset(tf.keras.callbacks.Callback):
    def __init__(self, *args, name='extra_val', verbose=None, **kwargs):
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
