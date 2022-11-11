import argparse

from ecoroar.util import generate_experiment_id

parser = argparse.ArgumentParser()
parser.add_argument('--seed',
                    action='store',
                    default=0,
                    type=int,
                    help='Random seed')
parser.add_argument('--dataset',
                    action='store',
                    default='IMDB',
                    type=str,
                    help='The dataset to fine-tune on')
parser.add_argument('--max-masking-ratio',
                    action='store',
                    default=0,
                    type=int,
                    help='The maximum masking ratio (percentage integer) to apply on the training dataset')

if __name__ == '__main__':
    args = parser.parse_args()

    experiment_id = generate_experiment_id(
        'masking',
        dataset=args.dataset, seed=args.seed, max_masking_ratio=args.max_masking_ratio
    )
    print(experiment_id)
