import argparse
import os.path as path

from datasets import load_dataset
from transformers import BertTokenizerFast, TFBertForSequenceClassification

thisdir = path.dirname(path.realpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--persistent-dir',
                    action='store',
                    default=path.realpath(path.join(thisdir, '..')),
                    type=str,
                    help='Directory where all persistent data will be stored')

if __name__ == "__main__":
    args = parser.parse_args()

    load_dataset("imdb", cache_dir=f'{args.persistent_dir}/cache/datasets')
    BertTokenizerFast.from_pretrained("bert-base-cased", cache_dir=f'{args.persistent_dir}/cache/tokenizer')
    TFBertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2, cache_dir=f'{args.persistent_dir}/cache/transformers')
