import pathlib
import argparse
import os.path as path

from ecoroar.util import model_name_to_huggingface_repo
from ecoroar.dataset import datasets
from ecoroar.tokenizer import HuggingfaceTokenizer
from ecoroar.model import HuggingfaceModel

thisdir = path.dirname(path.realpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--persistent-dir',
                    action='store',
                    default=pathlib.Path(__file__).absolute().parent.parent,
                    type=pathlib.Path,
                    help='Directory where all persistent data will be stored')

if __name__ == "__main__":
    args = parser.parse_args()

    for name, Dataset in datasets.items():
        dataset = Dataset(persistent_dir=args.persistent_dir)
        dataset.download()

    for model_name in [
        'roberta-sb', 'roberta-sl',
        'roberta-m15', 'roberta-m20', 'roberta-m30', 'roberta-m40',
        'roberta-m50', 'roberta-m60', 'roberta-m70', 'roberta-m80'
    ]:
        repo_name = model_name_to_huggingface_repo(model_name)
        tokenizer = HuggingfaceTokenizer(repo_name, persistent_dir=args.persistent_dir)
        model = HuggingfaceModel(repo_name, persistent_dir=args.persistent_dir, num_classes=2)
