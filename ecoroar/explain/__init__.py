
__all__ = ['RandomExplainer',
           'GradientL2Explainer', 'GradientL1Explainer',
           'InputTimesGradientSignExplainer', 'InputTimesGradientAbsExplainer',
           'IntegratedGradientSignExplainer', 'IntegratedGradientAbsExplainer',
           'explainers']

from .gradient import GradientL2Explainer, GradientL1Explainer
from .input_times_gradient import InputTimesGradientSignExplainer, InputTimesGradientAbsExplainer
from .integrated_gradient import IntegratedGradientSignExplainer, IntegratedGradientAbsExplainer
from .random import RandomExplainer

explainers = {
    Explainer._name: Explainer
    for Explainer
    in [RandomExplainer,
        GradientL2Explainer, GradientL1Explainer,
        InputTimesGradientSignExplainer, InputTimesGradientAbsExplainer,
        IntegratedGradientSignExplainer, IntegratedGradientAbsExplainer]
}
