
import json
import argparse
import os
import pathlib

from tqdm import tqdm
import pandas as pd

from ecoroar.dataset import datasets
from ecoroar.plot import bootstrap_confint, annotation


def tex_format_time(secs):
    hh, mm = divmod(secs // 60, 60)
    return f'{int(hh):02d}:{int(mm):02d}'


parser = argparse.ArgumentParser(
    description='Plots the 0% masking test performance given different training masking ratios'
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
    args, unknown = parser.parse_known_args()

    output_name = 'walltime_ood'

    if args.stage in ['both', 'preprocess']:

        # Read JSON files into dataframe
        results = []
        files = sorted((args.persistent_dir / 'results' / 'ood').glob('ood_*.json'))
        for file in tqdm(files, desc='Loading ood .json files'):
            with open(file, 'r') as fp:
                try:
                    data = json.load(fp)
                except json.decoder.JSONDecodeError:
                    print(f'{file} has a format error')

                if data['args']['model'] in ['roberta-sb', 'roberta-sl'] and \
                   data['args']['dataset'] in args.datasets:
                    results.append(data)

        df = pd.json_normalize(results)

    if args.stage in ['preprocess']:
        os.makedirs(args.persistent_dir / 'pandas', exist_ok=True)
        df.to_parquet((args.persistent_dir / 'pandas' / output_name).with_suffix('.parquet'))
    elif args.stage in ['plot']:
        df = pd.read_parquet((args.persistent_dir / 'pandas' / output_name).with_suffix('.parquet'))

    if args.stage in ['both', 'plot']:
        df_tab = (df
                  .groupby(['args.model', 'args.dataset'], group_keys=True)
                  .apply(bootstrap_confint(['durations.measure']))
                  .reset_index()
                  .pivot(index=['args.dataset'], columns=['args.model'], values=['durations.measure_mean']))

        df_tab_total = df_tab.sum(axis=0)

        os.makedirs(args.persistent_dir / 'tables' / args.format, exist_ok=True)
        with open(args.persistent_dir / 'tables' / args.format / f'{output_name}.tex', 'w') as fp:
            print(r'\begin{tabular}[t]{lcc}', file=fp)
            print(r'\toprule', file=fp)
            print(r'Dataset & \multicolumn{2}{c}{Walltime [hh:mm]} \\', file=fp)
            print(r'\cmidrule(r){2-3}', file=fp)
            print(r'& RoBERTa- & RoBERTa- \\', file=fp)
            print(r'& base & large \\', file=fp)
            print(r'\midrule', file=fp)
            first_row = True
            for dataset_name, dur in df_tab.iterrows():
                roberta_sb = tex_format_time(dur.loc["durations.measure_mean", :].loc["roberta-sb"])
                roberta_sl = tex_format_time(dur.loc["durations.measure_mean", :].loc["roberta-sl"])
                print(f'{annotation.dataset.labeller(dataset_name)} & {roberta_sb} & {roberta_sl} \\\\', file=fp)

            print(r'\midrule', file=fp)
            print(r'\midrule', file=fp)

            roberta_sb = df_tab_total.loc["durations.measure_mean", :].loc["roberta-sb"]
            roberta_sl = df_tab_total.loc["durations.measure_mean", :].loc["roberta-sl"]
            print(f'sum & {tex_format_time(roberta_sb)} & {tex_format_time(roberta_sl)} \\\\', file=fp)
            print(f'x5 seeds & {tex_format_time(roberta_sb * 5)} & {tex_format_time(roberta_sl * 5)} \\\\', file=fp)
            print(r'\bottomrule', file=fp)
            print(r'\end{tabular}', file=fp)
