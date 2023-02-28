
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

class Model(_AnnotationMapping):
    mapping = {
        'roberta-sb': 'RoBERTa base',
        'roberta-sl': 'RoBERTa large'
    }

class MaskingStrategy(_AnnotationMapping):
    mapping = {
        'goal': '0% masking',
        'uni': 'U[0%, 100%] masking',
        'half-ran': 'Sample 50/50',
        'half-det': 'Use 50/50'
    }

class Explainer(_AnnotationMapping):
    mapping = {
        'rand': 'Random',
        'grad-l2': 'Gradient (L2-norm)',
        'grad-l1': 'Gradient (L1-norm)',
        'inp-grad-sign': 'Input-times-gradient (signed)',
        'inp-grad-abs': 'Input-times-gradient (absolute)',
        'int-grad-sign': 'Integrated gradient (signed)',
        'int-grad-abs': 'Integrated gradient (absolute)',
    }

class AllAnnotations(_Singleton):
    def __init__(self) -> None:
        self.model = Model()
        self.masking_strategy = MaskingStrategy()
        self.explainer = Explainer()

annotation = AllAnnotations()
