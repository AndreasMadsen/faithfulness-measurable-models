
__all__ = ['GradientExplainer', 'InputTimesGradientExplainer', 'IntegratedGradientExplainer', 'explainers']

from .gradient import GradientExplainer
from .input_times_gradient import InputTimesGradientExplainer
from .integrated_gradient import IntegratedGradientExplainer
from .random import RandomExplainer

explainers = {
    Explainer._name: Explainer
    for Explainer
    in [RandomExplainer, GradientExplainer, InputTimesGradientExplainer, IntegratedGradientExplainer]
}
