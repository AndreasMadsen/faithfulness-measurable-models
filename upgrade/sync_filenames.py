
import pathlib
import argparse
import json
import shutil
from functools import partial

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from ecoroar.util import default_max_epochs, default_recursive, generate_experiment_id

parser = argparse.ArgumentParser()
parser.add_argument('--persistent-dir',
                    action='store',
                    default=pathlib.Path(__file__).absolute().parent.parent,
                    type=pathlib.Path,
                    help='Directory where all persistent data will be stored')
parser.add_argument('--dry',
                    action=argparse.BooleanOptionalAction,
                    default=True,
                    help='Only check. Don\'t remove anything')
parser.add_argument('--verbose',
                    action=argparse.BooleanOptionalAction,
                    default=None,
                    help='Print renamed files')
parser.add_argument('--max-workers',
                    action='store',
                    type=int,
                    default=4,
                    help='Maximum number of workers')
parser.add_argument('--chunksize',
                    action='store',
                    type=int,
                    default=10,
                    help='Chunksize for parrallel execution')

experiment_id_keys = set((
    'model', 'dataset', 'seed', 'max_epochs',
    'max_masking_ratio', 'masking_strategy', 'validation_dataset',
    'explainer', 'recursive', 'split', 'ood', 'dist_repeats'
))

def upgrade_result_file(args, result_file):
        experiment_type = result_file.parent.name
        experiment_id_old = result_file.stem

        with result_file.open('r') as fp:
            result = json.load(fp)
            result_args = result['args']
            result_namespace = argparse.Namespace(**result_args)

        if 'dataset' in result_args and 'max_epochs' not in result_args:
            result_namespace.max_epochs = None
            result_args['max_epochs'] = default_max_epochs(result_namespace)
        if 'explainer' in result_args and 'recursive' not in result_args:
            result_namespace.recursive = None
            result_args['recursive'] = default_recursive(result_namespace)

        experiment_id_new = generate_experiment_id(
            experiment_type,
            **{k: result_args[k] for k in result_args.keys() & experiment_id_keys}
        )

        if experiment_id_new == experiment_id_old:
            return

        if args.verbose:
            tqdm.write(f'{result_file.name} -> {result_file.with_stem(experiment_id_new).name}')
        if not args.dry:
            # The results also needs to be updated, to syncronize the ['args'] object
            with (result_file.with_stem(experiment_id_new)).open('w') as fp:
                json.dump(result, fp)
            result_file.unlink()

        if experiment_type in {'ood', 'faithfulness'}:
            intermediate_dir = args.persistent_dir / 'intermediate' / experiment_type
            for masking_ratio in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                intermediate_old = (intermediate_dir / experiment_id_old).with_suffix(f'.{masking_ratio}.tfds')
                intermediate_new = (intermediate_dir / experiment_id_new).with_suffix(f'.{masking_ratio}.tfds')
                if intermediate_old.exists() and intermediate_old.name != intermediate_new.name:
                    if args.verbose:
                        tqdm.write(f'  {intermediate_old.name} -> {intermediate_new.name}')
                    if not args.dry:
                        shutil.move(intermediate_old, intermediate_new)

if __name__ == '__main__':
    args, unknown = parser.parse_known_args()

    process_map(
        partial(upgrade_result_file, args),
        list((args.persistent_dir / 'results').glob('**/*.json')),
        max_workers=args.max_workers,
        chunksize=args.chunksize,
        desc='Renaming files')
