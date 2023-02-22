
__all__ = ['GradientExplainer', 'InputTimesGradientExplainer', 'explainers']

from .gradient import GradientExplainer
from .input_times_gradient import InputTimesGradientExplainer

explainers = {
    Explainer._name: Explainer
    for Explainer
    in [GradientExplainer, InputTimesGradientExplainer]
}
