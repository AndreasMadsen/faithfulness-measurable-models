
__all__ = ['IMDBDataset', 'MultiNLIDataset', 'datasets']

from .imdb import IMDBDataset
from .multi_nli import MultiNLIDataset

datasets = {
    Dataset._name: Dataset for Dataset in [IMDBDataset, MultiNLIDataset]
}
