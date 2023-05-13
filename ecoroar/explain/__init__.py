
__all__ = ['RandomExplainer',
           'GradientL2Explainer', 'GradientL1Explainer',
           'InputTimesGradientSignExplainer', 'InputTimesGradientAbsExplainer',
           'IntegratedGradientSignExplainer', 'IntegratedGradientAbsExplainer',
           'LeaveOneOutSign', 'LeaveOneOutAbs',
           'explainers']

from .gradient import GradientL2Explainer, GradientL1Explainer
from .input_times_gradient import InputTimesGradientSignExplainer, InputTimesGradientAbsExplainer
from .integrated_gradient import IntegratedGradientSignExplainer, IntegratedGradientAbsExplainer
from .random import RandomExplainer
from .leave_one_out import LeaveOneOutSign, LeaveOneOutAbs
from .beam_search import BeamSearch

explainers = {
    Explainer._name: Explainer
    for Explainer
    in [RandomExplainer,
        GradientL2Explainer, GradientL1Explainer,
        InputTimesGradientSignExplainer, InputTimesGradientAbsExplainer,
        IntegratedGradientSignExplainer, IntegratedGradientAbsExplainer,
        LeaveOneOutSign, LeaveOneOutAbs,
        BeamSearch]
}
