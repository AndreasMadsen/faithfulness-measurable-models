
import pathlib
import argparse
import shutil

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--persistent-dir',
                    action='store',
                    default=pathlib.Path(__file__).absolute().parent.parent,
                    type=pathlib.Path,
                    help='Directory where all persistent data will be stored')
parser.add_argument('--dry',
                    action=argparse.BooleanOptionalAction,
                    default=None,
                    help='Only check. Don\'t remove anything')
parser.add_argument('--verbose',
                    action=argparse.BooleanOptionalAction,
                    default=None,
                    help='Print removed files')

if __name__ == '__main__':
    args, unknown = parser.parse_known_args()

    for result_file in tqdm((args.persistent_dir / 'results').glob('masking_*r-100*.json'), desc='Removing result files'):
        checkpoint_file = args.persistent_dir / 'checkpoints' / result_file.stem / 'config.json'
        if not checkpoint_file.exists():
            if args.verbose:
                tqdm.write(str(result_file))
            if not args.dry:
                result_file.unlink(missing_ok=True)

    for checkpoint_dir in tqdm((args.persistent_dir / 'checkpoints').iterdir(), desc='Remove lone checkpoints'):
        result_file = (args.persistent_dir / 'results' / checkpoint_dir.stem).with_suffix('.json')
        if not result_file.exists():
            if args.verbose:
                tqdm.write(str(checkpoint_dir))
            if not args.dry:
                shutil.rmtree(checkpoint_dir, ignore_errors=True)

    for tensorboard_dir in tqdm((args.persistent_dir / 'tensorboard').iterdir(), desc='Remove lone tensorboard'):
        result_file = (args.persistent_dir / 'results' / tensorboard_dir.stem).with_suffix('.json')
        if not result_file.exists():
            if args.verbose:
                tqdm.write(str(tensorboard_dir))
            if not args.dry:
                shutil.rmtree(tensorboard_dir, ignore_errors=True)
