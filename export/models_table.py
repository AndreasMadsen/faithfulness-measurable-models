
import json
import argparse
import os
import pathlib

from tqdm import tqdm
import numpy as np
import pandas as pd

from ecoroar.dataset import datasets
from ecoroar.plot import bootstrap_confint, ci_formatter, annotation
from ecoroar.util import default_max_epochs

def select_target_metric(df):
    idx, cols = pd.factorize('results.' + df.loc[:, 'target_metric'])
    return df.assign(
        metric = df.reindex(cols, axis=1).to_numpy()[np.arange(len(df)), idx]
    )

parser = argparse.ArgumentParser(
    description = 'Plots the 0% masking test performance given different training masking ratios'
)
parser.add_argument('--persistent-dir',
                    action='store',
                    default=pathlib.Path(__file__).absolute().parent.parent,
                    type=pathlib.Path,
                    help='Directory where all persistent data will be stored')
parser.add_argument('--stage',
                    action='store',
                    default='both',
                    type=str,
                    choices=['preprocess', 'plot', 'both'],
                    help='Which export stage should be performed. Mostly just useful for debugging.')
parser.add_argument('--format',
                    action='store',
                    default='appendix',
                    type=str,
                    choices=['appendix'],
                    help='The dimentions and format of the table.')
parser.add_argument('--datasets',
                    action='store',
                    nargs='+',
                    default=list(datasets.keys()),
                    choices=datasets.keys(),
                    type=str,
                    help='The datasets to plot')
parser.add_argument('--performance-metric',
                    action='store',
                    default='primary',
                    type=str,
                    choices=['primary', 'loss', 'accuracy'],
                    help='Which metric to use as a performance metric.')

if __name__ == "__main__":
    #pd.set_option('display.max_rows', None)
    args, unknown = parser.parse_known_args()

    dataset_mapping = pd.DataFrame([
        {
            'args.dataset': dataset_name,
            'target_metric': datasets[dataset_name]._early_stopping_metric if args.performance_metric == 'primary' else args.performance_metric,
            'max_epoch': default_max_epochs(argparse.Namespace(dataset=dataset_name, max_epochs=None))
        }
        for dataset_name in args.datasets
    ])
    output_name = 'models'

    if args.stage in ['both', 'preprocess']:

        # Read JSON files into dataframe
        results = []
        files = sorted((args.persistent_dir / 'results' / 'masking').glob('masking_*.json'))
        for file in tqdm(files, desc='Loading masking .json files'):
            with open(file, 'r') as fp:
                try:
                    data = json.load(fp)
                except json.decoder.JSONDecodeError:
                    print(f'{file} has a format error')

                if data['args']['max_masking_ratio'] == 0 and \
                   data['args']['model'] in ['roberta-sb', 'roberta-sl'] and \
                   data['args']['dataset'] in args.datasets and \
                   data['args']['validation_dataset'] == 'nomask':
                    results.append(data)

        df = pd.json_normalize(results).explode('results', ignore_index=True)
        results = pd.json_normalize(df.pop('results')).add_prefix('results.')
        df = pd.concat([df, results], axis=1)

        # Select test metric
        df = (df
              .merge(dataset_mapping, on='args.dataset')
              .transform(select_target_metric)
              .query('`results.masking_ratio` == 0'))

    if args.stage in ['preprocess']:
        os.makedirs(args.persistent_dir / 'pandas', exist_ok=True)
        df.to_parquet((args.persistent_dir / 'pandas' / output_name).with_suffix('.parquet'))
    elif args.stage in ['plot']:
        df = pd.read_parquet((args.persistent_dir / 'pandas' / output_name).with_suffix('.parquet'))

    if args.stage in ['both', 'plot']:
        df_tab = (df
                .groupby(['args.model', 'args.dataset', 'max_epoch'], group_keys=True)
                .apply(bootstrap_confint(['metric']))
                .reset_index()
                .transform(ci_formatter(['metric']))
                .pivot(index=['args.dataset', 'max_epoch'], columns=['args.model'], values=['metric_format']))

        os.makedirs(args.persistent_dir / 'tables' / args.format, exist_ok=True)
        with open(args.persistent_dir / 'tables' / args.format / f'{output_name}.tex', 'w') as fp:
            print(r'\begin{tabular}{lccc}', file=fp)
            print(r'\toprule', file=fp)
            print(r'Dataset & max epoch & \multicolumn{2}{c}{Performance} \\', file=fp)
            print(r'\cmidrule(r){3-4}', file=fp)
            print(r'& & RoBERTa- & RoBERTa- \\', file=fp)
            print(r'& & base & large \\', file=fp)
            print(r'\midrule', file=fp)
            for (dataset_name, max_epoch), perf in df_tab.iterrows():
                roberta_sb = perf.loc["metric_format", :].loc["roberta-sb"]
                roberta_sl = perf.loc["metric_format", :].loc["roberta-sl"]

                print(f'{annotation.dataset.labeller(dataset_name)} & {max_epoch} & {roberta_sb} & {roberta_sl} \\\\', file=fp)
            print(r'\bottomrule', file=fp)
            print(r'\end{tabular}', file=fp)
