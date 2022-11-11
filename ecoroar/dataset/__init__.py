
__all__ = ['BoolQDataset', 'CoLADataset', 'IMDBDataset',
           'MNLIDataset', 'QQPDataset', 'SST2Dataset',
           'datasets']

from .boolq import BoolQDataset
from .cola import CoLADataset
from .imdb import IMDBDataset
from .mnli import MNLIDataset
from .qqp import QQPDataset
from .sst2 import SST2Dataset

datasets = {
    Dataset._name: Dataset
    for Dataset
    in [BoolQDataset, CoLADataset, IMDBDataset, MNLIDataset, QQPDataset, SST2Dataset]
}
