
from functools import cached_property

class _AnnotationMapping(dict):
    @cached_property
    def breaks(self):
        return list(self.keys())

    @cached_property
    def labels(self):
        return list(self.values())

    def labeller(self, key):
        return self.get(key, key)

    def __or__(self, other):
        return _AnnotationMapping(dict.__or__(self, other))

    def __ror__(self, other):
        return _AnnotationMapping(dict.__ror__(self, other))

    def __ior__(self, other):
        return super.__ior__(self, other)

class _AllAnnotations():
    def __init__(self) -> None:
        self.dataset = _AnnotationMapping({
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
        })
        self.model = _AnnotationMapping({
            'roberta-sb': 'RoBERTa base',
            'roberta-sl': 'RoBERTa large'
        })
        self.masking_strategy = _AnnotationMapping({
            'goal': 'No masking',
            'uni': 'Maksing',
            'half-det': 'Use 50/50'
        })
        self.max_masking_ratio = _AnnotationMapping({
            '0': 'Plain fine-tune',
            '100': 'Masked fine-tune',
        })
        self.validation_dataset = _AnnotationMapping({
            'nomask': 'No masking',
            'mask': 'Maksing',
            'both': 'Use both'
        })
        self.validation = _AnnotationMapping({
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
        })
        self.explainer = _AnnotationMapping({
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
        })
        self.explainer_sign = _AnnotationMapping({
            'abs': 'Absolute',
            'sign': 'Signed'
        })
        self.explainer_base = _AnnotationMapping({
            'rand': 'Random',
            'grad-l2': 'Grad ($L_2$)',
            'grad-l1': 'Grad ($L_1$)',
            'inp-grad': 'x $\odot$ grad',
            'int-grad': 'IG',
            'loo': 'LOO',
            'beam-10': 'Beam (10)',
        })


annotation = _AllAnnotations()
