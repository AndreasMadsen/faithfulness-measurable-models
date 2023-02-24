
__all__ = ['GradientExplainer', 'InputTimesGradientExplainer', 'IntegratedGradientExplainer', 'explainers']

from .gradient import GradientExplainer
from .input_times_gradient import InputTimesGradientExplainer
from .integrated_gradient import IntegratedGradientExplainer

explainers = {
    Explainer._name: Explainer
    for Explainer
    in [GradientExplainer, InputTimesGradientExplainer, IntegratedGradientExplainer]
}
