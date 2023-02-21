
import pathlib
from contextlib import contextmanager

import transformers
from transformers import AutoConfig

from ..types import Model
from .roberta import ExtraRoBERTaForSequenceClassification

@contextmanager
def silence_huggingface():
    level = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    yield
    transformers.logging.set_verbosity(level)


def HuggingfaceModel(model_name: str, persistent_dir: pathlib.Path, num_classes: int) -> Model:
    """Creates a sequence classification model

    Args:
        model_name (str): the model name as input to Model
        persistent_dir (str): Used to store the downloaded tokenizer
        num_classes (int): The number of output labels/classes

    Returns:
        Model: A SequenceClassification instance
    """
    config = AutoConfig.from_pretrained(model_name,
                                        cache_dir=persistent_dir / 'cache' / 'transformers')

    match config.model_type:
        case 'roberta':
            with silence_huggingface():
                return ExtraRoBERTaForSequenceClassification.from_pretrained(
                    model_name,
                    num_classes=num_classes,
                    cache_dir=persistent_dir / 'cache' / 'transformers'
                )

    raise NotImplementedError(f'An embedding abstraction have been implemented for a {config.model_type} model')
