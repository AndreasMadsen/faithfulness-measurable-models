import argparse
import os.path as path

from transformers import TFBertForSequenceClassification
from ecoroar.dataset import IMDBDataset
from ecoroar.tokenizer import BertTokenizer

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
    model = TFBertForSequenceClassification.from_pretrained('bert-base-cased',
        num_labels=dataset.num_classes, cache_dir=f'{args.persistent_dir}/download/transformers')
