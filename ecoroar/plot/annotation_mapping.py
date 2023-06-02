
from abc import ABCMeta, abstractmethod
from functools import cached_property

class _Singleton:
    # make these objects singletons
    _singleton = None
    def __new__(cls):
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cls._singleton

class _AnnotationMapping(_Singleton, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def mapping(self):
        pass

    @cached_property
    def breaks(self):
        return list(self.mapping.keys())

    @cached_property
    def labels(self):
        return list(self.mapping.values())

    def labeller(self, key):
        return self.mapping.get(key, key)

class Dataset(_AnnotationMapping):
    mapping = {
        'bAbI-1': 'bAbI-1',
        'bAbI-2': 'bAbI-2',
        'bAbI-3': 'bAbI-3',
        'BoolQ': 'BoolQ',
        'CB': 'CB',
        'CoLA': 'CoLA',
        'MIMIC-a': 'Anemia',
        'MIMIC-d': 'Diabetes',
        'MRPC': 'MRPC',
        'RTE': 'RTE',
        'SST2': 'SST2',
        'IMDB': 'IMDB',
        'MNLI': 'MNLI',
        'QNLI': 'QNLI',
        'QQP': 'QQP',
        'WNLI': 'WNLI'
    }

class Model(_AnnotationMapping):
    mapping = {
        'roberta-sb': 'RoBERTa base',
        'roberta-sl': 'RoBERTa large'
    }

class MaskingStrategy(_AnnotationMapping):
    mapping = {
        'goal': 'No masking',
        'uni': 'Maksing',
        'half-det': 'Use 50/50'
    }

class ValidationDataset(_AnnotationMapping):
    mapping = {
        'nomask': 'No masking',
        'mask': 'Maksing',
        'both': 'Use both'
    }

class Explainer(_AnnotationMapping):
    mapping = {
        'rand': 'Random',
        'grad-l2': 'Grad ($L_2$)',
        'grad-l1': 'Grad ($L_1$)',
        'inp-grad-sign': 'x $\odot$ grad (sign)',
        'inp-grad-abs': 'x $\odot$ grad (abs)',
        'int-grad-sign': 'IG (sign)',
        'int-grad-abs': 'IG (abs)',
        'loo-sign': 'LOO (sign)',
        'loo-abs': 'LOO (abs)',
        'beam-sign-10': 'Beam (10)',
    }

class Validation(_AnnotationMapping):
    mapping = {
        'metric.val': 'Earily stopping',
        'metric.val_0': '0% masking',
        'metric.val_10': '10% masking',
        'metric.val_20': '20% masking',
        'metric.val_30': '30% masking',
        'metric.val_40': '40% masking',
        'metric.val_50': '50% masking',
        'metric.val_60': '60% masking',
        'metric.val_70': '70% masking',
        'metric.val_80': '80% masking',
        'metric.val_90': '90% masking',
        'metric.val_100': '100% masking',
    }

class AllAnnotations(_Singleton):
    def __init__(self) -> None:
        self.dataset = Dataset()
        self.model = Model()
        self.masking_strategy = MaskingStrategy()
        self.explainer = Explainer()
        self.validation = Validation()
        self.validation_dataset = ValidationDataset()

annotation = AllAnnotations()
