import argparse
import os.path as path

from transformers import TFAutoModel
from ecoroar.dataset import IMDBDataset
from ecoroar.tokenizer import BertTokenizer, HuggingfaceTokenizer

thisdir = path.dirname(path.realpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--persistent-dir',
                    action='store',
                    default=path.realpath(path.join(thisdir, '..')),
                    type=str,
                    help='Directory where all persistent data will be stored')

if __name__ == "__main__":
    args = parser.parse_args()

    dataset = IMDBDataset(persistent_dir=args.persistent_dir)
    dataset.download()

    tokenizer = BertTokenizer('bert-base-cased', persistent_dir=args.persistent_dir)

    for model_name in ['bert-base-cased', 'roberta-base']:
        tokenizer = HuggingfaceTokenizer(model_name, persistent_dir=args.persistent_dir)
        model = TFAutoModel.from_pretrained(
            model_name,
            cache_dir=f'{args.persistent_dir}/download/transformers'
        )
