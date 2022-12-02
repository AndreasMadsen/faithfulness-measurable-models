
import pathlib
import argparse

from tqdm import tqdm
import pandas as pd
import plotnine as p9
import numpy as np

import tensorflow as tf
from ecoroar.util import model_name_to_huggingface_repo
from ecoroar.dataset import datasets
from ecoroar.tokenizer import HuggingfaceTokenizer
from ecoroar.model import HuggingfaceModel

parser = argparse.ArgumentParser()
parser.add_argument('--persistent-dir',
                    action='store',
                    default=pathlib.Path(__file__).absolute().parent.parent,
                    type=pathlib.Path,
                    help='Directory where all persistent data will be stored')
parser.add_argument('--seed',
                    action='store',
                    default=0,
                    type=int,
                    help='Random seed')
parser.add_argument('--model',
                    action='store',
                    default='roberta-sb',
                    type=str,
                    help='Model name')
parser.add_argument('--huggingface-repo',
                    action='store',
                    default=None,
                    type=str,
                    help='Valid huggingface repo')
parser.add_argument('--dataset',
                    action='store',
                    default='IMDB',
                    type=str,
                    help='The dataset to fine-tune on')
parser.add_argument('--batch-size',
                    action='store',
                    default=16,
                    type=int,
                    help='The batch size to use for training and evaluation')

if __name__ == '__main__':
    args = parser.parse_args()

    if args.huggingface_repo is None:
        args.huggingface_repo = model_name_to_huggingface_repo(args.model)

    tf.keras.utils.set_random_seed(args.seed)

    tokenizer = HuggingfaceTokenizer(args.huggingface_repo, persistent_dir=args.persistent_dir)
    dataset = datasets[args.dataset](persistent_dir=args.persistent_dir, seed=args.seed)

    dataset_train = dataset.train(tokenizer) \
        .shuffle(dataset.train_num_examples, seed=args.seed) \
        .padded_batch(args.batch_size, padding_values=(tokenizer.padding_values, None)) \
        .prefetch(tf.data.AUTOTUNE)

    dataset_valid = dataset.valid(tokenizer) \
    #    .padded_batch(args.batch_size, padding_values=(tokenizer.padding_values, None)) \
        .prefetch(tf.data.AUTOTUNE)

    dataset_test = dataset.test(tokenizer) \
    #    .padded_batch(args.batch_size, padding_values=(tokenizer.padding_values, None)) \
        .prefetch(tf.data.AUTOTUNE)

    dataset_batch_length = []
    for split, dataset in [('train', dataset_train), ('valid', dataset_valid), ('test', dataset_test)]:
        for x, y in tqdm(dataset, desc=f'Collecting {split} split'):
            dataset_batch_length.append({
                'split': split,
                'batch_length': x['input_ids'].shape[1]
            })

    df = pd.DataFrame(dataset_batch_length)

    lengths = df.loc[df['split'] == 'train', 'batch_length'].to_numpy()
    print(np.quantile(lengths, q=[0.25, 0.5, 0.75, 0.9, 1.0]))
    print('maximum length:', df['batch_length'].max())

    p = (
        p9.ggplot(df, p9.aes(x='batch_length', fill='split'))
        + p9.geom_histogram()
        + p9.facet_wrap('split', ncol=1)
    )
    print(p)



