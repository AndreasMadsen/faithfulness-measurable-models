
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

if __name__ == '__main__':
    args, unknown = parser.parse_known_args()

    for result_file in tqdm((args.persistent_dir / 'results').iterdir(), desc='Removing result files'):
        if not result_file.stem.startswith('masking'):
            continue

        checkpoint_file = args.persistent_dir / 'checkpoints' / result_file.stem / 'config.json'
        if not checkpoint_file.exists():
            result_file.unlink(missing_ok=True)

    for checkpoint_dir in tqdm((args.persistent_dir / 'checkpoints').iterdir(), desc='Remove lone checkpoints'):
        result_file = (args.persistent_dir / 'results' / checkpoint_dir.stem).with_suffix('.json')
        if not result_file.exists():
            shutil.rmtree(checkpoint_dir, ignore_errors=True)

    for tensorboard_dir in tqdm((args.persistent_dir / 'tensorboard').iterdir(), desc='Remove lone tensorboard'):
        result_file = (args.persistent_dir / 'results' / tensorboard_dir.stem).with_suffix('.json')
        if not result_file.exists():
            shutil.rmtree(tensorboard_dir, ignore_errors=True)
