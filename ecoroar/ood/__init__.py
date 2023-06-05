__all__ = ['ood_detectors', 'MaSF', 'MaSFSlow']

from .masf import MaSF
from .masf_slow import MaSFSlow

ood_detectors = {
    OODDetector._name: OODDetector
    for OODDetector
    in [MaSF, MaSFSlow]
}
