import pathlib
import argparse
import os.path as path

from tqdm import tqdm

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

    for name, Dataset in (pbar := tqdm(datasets.items())):
        pbar.set_description(f'Downloading dataset {name}')
        pbar.set_description(f'Downloading dataset {name}')
        dataset = Dataset(persistent_dir=args.persistent_dir)
        dataset.download()

    models = [
        'roberta-sb', 'roberta-sl',
        'roberta-m15', 'roberta-m20', 'roberta-m30', 'roberta-m40',
        'roberta-m50', 'roberta-m60', 'roberta-m70', 'roberta-m80'
    ]

    for model_name in tqdm(pbar := tqdm(models)):
        pbar.set_description(f'Downloading model {model_name}')
        repo_name = model_name_to_huggingface_repo(model_name)
        tokenizer = HuggingfaceTokenizer(repo_name, persistent_dir=args.persistent_dir)
        model = HuggingfaceModel(repo_name, persistent_dir=args.persistent_dir, num_classes=2)
