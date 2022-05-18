
import pathlib

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
        return TFAutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            cache_dir=persistent_dir / 'cache' / 'transformers'
        )
