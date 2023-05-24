import pathlib

import argparse

from ecoroar.util import generate_experiment_id, default_max_epochs, default_recursive

parser = argparse.ArgumentParser()
parser.add_argument('scriptpath',
                    action='store',
                    type=pathlib.Path,
                    help='The script path that the job will execute')
parser.add_argument('--seed',
                    action='store',
                    default=None,
                    type=int,
                    help='Random seed')
parser.add_argument('--model',
                    action='store',
                    default=None,
                    type=str,
                    help='Model name')
parser.add_argument('--dataset',
                    action='store',
                    default=None,
                    type=str,
                    help='The dataset to fine-tune on')
parser.add_argument('--max-epochs',
                    action='store',
                    default=None,
                    type=int,
                    help='The max number of epochs to use')
parser.add_argument('--max-masking-ratio',
                    action='store',
                    default=None,
                    type=int,
                    help='The maximum masking ratio (percentage integer) to apply on the training dataset')
parser.add_argument('--masking-strategy',
                    default=None,
                    choices=['uni', 'half-det', 'half-ran'],
                    type=str,
                    help='The masking strategy to use for masking during fune-tuning')
parser.add_argument('--validation-dataset',
                    default=None,
                    choices=['nomask', 'mask', 'both'],
                    type=str,
                    help='The transformation applied to the validation dataset used for early stopping.')
parser.add_argument('--explainer',
                    default=None,
                    type=str,
                    help='The importance measure algorithm to use for explanation')
parser.add_argument('--recursive',
                    action=argparse.BooleanOptionalAction,
                    default=None,
                    type=bool,
                    help='Are the importance measures computed recursively.')
parser.add_argument('--split',
                    default=None,
                    choices=['train', 'valid', 'test'],
                    type=str,
                    help='The dataset split to evaluate faithfulness on')
parser.add_argument('--ood',
                    default=None,
                    choices=['masf', 'masf-slow'],
                    type=str,
                    help='The OOD detection method')
parser.add_argument('--dist-repeats',
                    default=None,
                    type=int,
                    help='The number of repeats used to estimate the distribution')


if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    args.max_epochs = default_max_epochs(args)
    args.recursive = default_recursive(args)

    experiment_id = generate_experiment_id(
        args.scriptpath.name.rstrip('.py'),
        model=args.model, dataset=args.dataset,
        seed=args.seed, max_epochs=args.max_epochs,
        max_masking_ratio=args.max_masking_ratio, masking_strategy=args.masking_strategy,
        validation_dataset=args.validation_dataset,
        explainer=args.explainer, recursive=args.recursive, split=args.split,
        ood=args.ood, dist_repeats=args.dist_repeats
    )
    print(experiment_id)
