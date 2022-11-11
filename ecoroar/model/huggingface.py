
import pathlib

import transformers
from transformers import TFAutoModelForSequenceClassification


class HuggingfaceModel(TFAutoModelForSequenceClassification):
    def __new__(cls, model_name: str, persistent_dir: pathlib.Path, num_classes: int) -> TFAutoModelForSequenceClassification:
        """Creates a sequence classification model

        Args:
            model_name (str): the model name as input to transformers.TFAutoModelForSequenceClassification
            persistent_dir (str): Used to store the downloaded tokenizer
            num_classes (int): The number of output labels/classes

        Returns:
            TFAutoModelForSequenceClassification: A SequenceClassification instance
        """
        level = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        try:
            model = TFAutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                cache_dir=persistent_dir / 'cache' / 'transformers'
            )
        finally:
            transformers.logging.set_verbosity(level)

        return model
