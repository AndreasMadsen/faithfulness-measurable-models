
import pathlib
import argparse
import json

from tqdm.contrib.concurrent import process_map
from functools import partial

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
                    default=False,
                    help='Print removed files')
parser.add_argument('--max-workers',
                    action='store',
                    type=int,
                    default=4,
                    help='Maximum number of workers')
parser.add_argument('--chunksize',
                    action='store',
                    type=int,
                    default=1,
                    help='Chunksize for parrallel execution')


def upgrade_result_file(args, result_file):
    with (args.persistent_dir / 'results' / 'ood' / result_file).open('r') as fp:
        old_results = json.load(fp)

    new_results = old_results.copy()
    new_results['args']['ood'] = old_results['args']['ood'].lower()

    if not args.dry:
        with (args.persistent_dir / 'results' / 'ood' / result_file).open('w') as fp:
            json.dump(new_results, fp)

if __name__ == '__main__':
    args, unknown = parser.parse_known_args()

    process_map(
        partial(upgrade_result_file, args),
        list((args.persistent_dir / 'results' / 'ood').glob('*.json')),
        max_workers=args.max_workers,
        chunksize=args.chunksize,
        desc='ood result files')
