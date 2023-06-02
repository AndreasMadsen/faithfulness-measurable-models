
import argparse
import os
import pathlib

import numpy as np
import pandas as pd

from ecoroar.dataset import datasets
from ecoroar.plot import annotation

def select_target_metric(df):
    idx, cols = pd.factorize('results.' + df.loc[:, 'metric'])
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

if __name__ == "__main__":
    #pd.set_option('display.max_rows', None)
    args, unknown = parser.parse_known_args()

    output_name = 'datasets'

    if args.stage in ['both', 'preprocess']:
        df = pd.DataFrame([
            datasets[dataset_name].summary()
            for dataset_name in args.datasets
        ])

    if args.stage in ['preprocess']:
        os.makedirs(args.persistent_dir / 'pandas', exist_ok=True)
        df.to_parquet((args.persistent_dir / 'pandas' / output_name).with_suffix('.parquet'))
    elif args.stage in ['plot']:
        df = pd.read_parquet((args.persistent_dir / 'pandas' / output_name).with_suffix('.parquet'))

    if args.stage in ['both', 'plot']:

        os.makedirs(args.persistent_dir / 'tables' / args.format, exist_ok=True)
        with open(args.persistent_dir / 'tables' / args.format / f'{output_name}.tex', 'w') as fp:
            print(r'\begin{tabular}{lcccccc}', file=fp)
            print(r'\toprule', file=fp)
            print(r'Dataset & \multicolumn{3}{c}{Size} & \multicolumn{2}{c}{Inputs} & Performance \\', file=fp)
            print(r'\cmidrule(r){2-4} \cmidrule(r){5-6}', file=fp)
            print(r'& Train & Validation & Test & masked & auxilary & class-majority \\', file=fp)
            print(r'\midrule', file=fp)
            for row_i, row in df.iterrows():
                if row["auxilary"] is None:
                    aux = "--"
                else:
                    aux = f'\\texttt{{{row["auxilary"]}}}'

                print(f'{annotation.dataset.labeller(row["name"])} & ${row["train"]}$ & ${row["valid"]}$ & ${row["test"]}$ & \\texttt{{{row["masked"]}}} & {aux} & ${row["baseline"]*100:.0f}\\%$ \\\\', file=fp)
            print(r'\bottomrule', file=fp)
            print(r'\end{tabular}', file=fp)
