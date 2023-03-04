
import pathlib
from contextlib import contextmanager

import transformers
from transformers import AutoConfig

from ..types import Model
from .roberta import TFRoBERTaForSequenceClassificationExtra, TFRoBERTaPreLayerNormForSequenceClassificationExtra

@contextmanager
def silence_huggingface():
    level = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    yield
    transformers.logging.set_verbosity(level)

def huggingface_constructor(config: AutoConfig):
    """Returns a sequence classification class

    Args:
        config (AutoConfig): huggingface config object

    Returns:
        Model: SequenceClassification class
    """
    match config.model_type:
        case 'roberta':
            return TFRoBERTaForSequenceClassificationExtra
        case 'roberta-prelayernorm':
            return TFRoBERTaPreLayerNormForSequenceClassificationExtra

    raise NotImplementedError(f'An embedding abstraction have been implemented for a {config.model_type} model')

def huggingface_model_from_repo(repo: str, persistent_dir: pathlib.Path, num_classes: int) -> Model:
    """Creates a sequence classification model

    Args:
        repo (str): the model name as input to Model
        persistent_dir (str): Used to store the downloaded tokenizer
        num_classes (int): The number of output labels/classes

    Returns:
        Model: A SequenceClassification instance
    """
    config = AutoConfig.from_pretrained(repo,
                                        cache_dir=persistent_dir / 'cache' / 'transformers')
    SequenceClassification = huggingface_constructor(config)

    with silence_huggingface():
        return SequenceClassification.from_pretrained(
            repo,
            num_labels=num_classes,
            cache_dir=persistent_dir / 'cache' / 'transformers'
        )

def huggingface_model_from_local(filepath: str) -> Model:
    """Creates a sequence classification model

    Args:
        filepath (str): the model name as input to Model

    Returns:
        Model: A SequenceClassification instance
    """
    config = AutoConfig.from_pretrained(filepath)
    SequenceClassification = huggingface_constructor(config)
    with silence_huggingface():
        return SequenceClassification.from_pretrained(
            filepath,
            local_files_only=True
        )
