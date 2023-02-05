
import json
import argparse
import os
import pathlib
import shutil

from tqdm import tqdm

from ecoroar.util import generate_experiment_id


parser = argparse.ArgumentParser(
    description = 'Convert results from high-masking version to alt-fine-tuning version'
)
parser.add_argument('--persistent-dir',
                    action='store',
                    default=pathlib.Path(__file__).absolute().parent.parent,
                    type=pathlib.Path,
                    help='Directory where all persistent data will be stored')

if __name__ == "__main__":
    args, unknown = parser.parse_known_args()

    files = sorted((args.persistent_dir / 'results').glob('masking_*.json'))
    for json_filepath in (pbar := tqdm(files)):
        old_experiment_id = json_filepath.stem
        pbar.set_description(f'Converting {old_experiment_id}', refresh=False)

        with open(json_filepath, 'r') as fp:
            try:
                results = json.load(fp)
            except json.decoder.JSONDecodeError:
                print(f'{json_filepath} has a format error')

        results['args']['masking_strategy'] = 'uni'

        new_experiment_id = generate_experiment_id(
            'masking',
            model=results['args']['model'], dataset=results['args']['dataset'],
            seed=results['args']['seed'], max_epochs=results['args']['max_epochs'],
            max_masking_ratio=results['args']['max_masking_ratio'], masking_strategy=results['args']['masking_strategy']
        )

        # Remove old results and save new converted results
        os.remove(json_filepath)
        with open(args.persistent_dir / 'results' / f'{new_experiment_id}.json', "w") as f:
            json.dump(results, f)

        # Move auxilary files
        if (args.persistent_dir / 'checkpoints' / old_experiment_id).exists():
            shutil.move(args.persistent_dir / 'checkpoints' / old_experiment_id, args.persistent_dir / 'checkpoints' / new_experiment_id)
        if (args.persistent_dir / 'tensorboard' / old_experiment_id).exists():
            shutil.move(args.persistent_dir / 'tensorboard' / old_experiment_id, args.persistent_dir / 'tensorboard' / new_experiment_id)
