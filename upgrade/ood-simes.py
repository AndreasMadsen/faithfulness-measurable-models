
import pathlib
import argparse
import json

import tensorflow as tf
from scipy.stats import chi2
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

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


if __name__ == '__main__':
    args, unknown = parser.parse_known_args()

    def upgrade_result_file(result_file):
        if all(
            (args.persistent_dir / 'intermediate' / 'ood' / result_file.stem).with_suffix(f'.{masking_ratio}.tfds').exists() \
            for masking_ratio in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        ):
            with (args.persistent_dir / 'results' / 'ood' / result_file).open('r') as fp:
                old_results = json.load(fp)

            # already converted
            if any('method' in count_result for count_result in old_results['results']):
                if args.verbose:
                    tqdm.write(f'{result_file.stem} already converted')
                return

            # Reformat existing results
            new_results = old_results.copy()
            new_results['results'] = [
                { 'method': 'proportion',
                  'value': count_result['proportion'],
                  'masking_ratio': count_result['masking_ratio'],
                  'threshold': count_result['threshold']
                }
                for count_result in old_results['results']
            ]

            # create simes statistics
            for masking_ratio in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                p_values_dataset = tf.data.Dataset.load(str(
                    (args.persistent_dir / 'intermediate' / 'ood' / result_file.stem).with_suffix(f'.{masking_ratio}.tfds')
                ))
                p_values_n_obs = p_values_dataset.reduce(
                    tf.constant(0, dtype=tf.dtypes.int64),
                    lambda count, p_values: count + tf.size(p_values, out_type=tf.dtypes.int64)
                )
                p_values = p_values_dataset \
                    .apply(lambda ds: ds.rebatch(p_values_n_obs)) \
                    .get_single_element()

                p_values_count = tf.size(p_values).numpy()

                simes_p_value = tf.math.reduce_min(
                    tf.cast(p_values_count / tf.range(1, p_values_count + 1), dtype=p_values.dtype) * \
                    tf.sort(p_values, direction='ASCENDING')
                ).numpy()

                fisher_statistic = (-2 * tf.math.reduce_sum(tf.math.log(p_values + 1e-8))).numpy()
                fisher_p_value = chi2.cdf(fisher_statistic, 2 * p_values_count)

                new_results['results'].append({
                    'method': 'simes',
                    'value': simes_p_value.tolist(),
                    'masking_ratio': masking_ratio / 100
                })
                new_results['results'].append({
                    'method': 'fisher',
                    'value': fisher_p_value.tolist(),
                    'masking_ratio': masking_ratio / 100
                })

                if args.verbose:
                    tqdm.write(f'{result_file.stem} at {masking_ratio}, simes: {simes_p_value}, fisher: {fisher_p_value}')

            if not args.dry:
                with (args.persistent_dir / 'results' / 'ood' / result_file).open('w') as fp:
                    json.dump(new_results, fp)

    process_map(
        upgrade_result_file,
        list((args.persistent_dir / 'results' / 'ood').glob('*.json')),
        max_workers=args.max_workers,
        chunksize=args.chunksize,
        desc='ood result files')
