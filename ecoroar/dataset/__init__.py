
__all__ = ['BoolQDataset', 'CBDataset', 'CoLADataset', 'IMDBDataset',
           'MNLIDataset', 'MRPCDataset', 'QNLIDataset', 'QQPDataset',
           'RTEDataset', 'SST2Dataset', 'WNLIDataset',
           'Babi1Dataset', 'Babi2Dataset', 'Babi3Dataset',
           'datasets']

from .boolq import BoolQDataset
from .cb import CBDataset
from .cola import CoLADataset
from .imdb import IMDBDataset
from .mnli import MNLIDataset
from .mrpc import MRPCDataset
from .qnli import QNLIDataset
from .qqp import QQPDataset
from .rte import RTEDataset
from .sst2 import SST2Dataset
from .wnli import WNLIDataset
from .babi import Babi1Dataset, Babi2Dataset, Babi3Dataset

datasets = {
    Dataset._name: Dataset
    for Dataset
    in [BoolQDataset, CBDataset, CoLADataset, IMDBDataset,
        MNLIDataset, MRPCDataset, QNLIDataset, QQPDataset,
        RTEDataset, SST2Dataset, WNLIDataset,
        Babi1Dataset, Babi2Dataset, Babi3Dataset]
}
