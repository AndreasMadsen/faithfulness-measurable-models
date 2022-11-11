import pathlib
import argparse
import os.path as path

from itertools import product
from tqdm import tqdm

from ecoroar.dataset import datasets
from ecoroar.tokenizer import HuggingfaceTokenizer

thisdir = path.dirname(path.realpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--persistent-dir',
                    action='store',
                    default=pathlib.Path(__file__).absolute().parent.parent,
                    type=pathlib.Path,
                    help='Directory where all persistent data will be stored')

if __name__ == "__main__":
    args = parser.parse_args()

    for model_name in ['roberta-base']:
        tokenizer = HuggingfaceTokenizer(model_name, persistent_dir=args.persistent_dir)

        for seed, (name, Dataset) in (pbar := tqdm(product(range(5), datasets.items()))):
            pbar.set_description(f'Processing {name}(seed={seed})')
            dataset = Dataset(persistent_dir=args.persistent_dir, seed=seed)
            dataset.preprocess(tokenizer)
