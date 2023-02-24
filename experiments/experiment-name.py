import pathlib

import argparse

from ecoroar.util import generate_experiment_id, default_max_epochs

parser = argparse.ArgumentParser()
parser.add_argument('scriptpath',
                    action='store',
                    type=pathlib.Path,
                    help='The script path that the job will execute')
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
parser.add_argument('--dataset',
                    action='store',
                    default='IMDB',
                    type=str,
                    help='The dataset to fine-tune on')
parser.add_argument('--max-epochs',
                    action='store',
                    default=None,
                    type=int,
                    help='The max number of epochs to use')
parser.add_argument('--max-masking-ratio',
                    action='store',
                    default=0,
                    type=int,
                    help='The maximum masking ratio (percentage integer) to apply on the training dataset')
parser.add_argument('--masking-strategy',
                    default='uni',
                    choices=['uni', 'half-det', 'half-ran'],
                    type=str,
                    help='The masking strategy to use for masking during fune-tuning')

if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    args.max_epochs = default_max_epochs(args)

    experiment_id = generate_experiment_id(
        args.scriptpath.name.rstrip('.py'),
        model=args.model, dataset=args.dataset,
        seed=args.seed, max_epochs=args.max_epochs,
        max_masking_ratio=args.max_masking_ratio, masking_strategy=args.masking_strategy
    )
    print(experiment_id)
