
import json
import argparse
import os
import pathlib
from functools import partial

from tqdm import tqdm
import numpy as np
import pandas as pd
import plotnine as p9
from mizani.palettes import brewer_pal

from ecoroar.dataset import datasets
from ecoroar.plot import bootstrap_confint, annotation
from ecoroar.util import generate_experiment_id


def select_target_metric(df, selectors=dict()):
    add_columns = dict()
    for new_column, prefix in selectors.items():
        idx, cols = pd.factorize(prefix + df.loc[:, 'target_metric'])
        add_columns[new_column] = df.reindex(cols, axis=1).to_numpy()[np.arange(len(df)), idx]

    return df.assign(**add_columns)


def delete_columns(df, prefix):
    remove_columns = df.columns[df.columns.str.startswith(prefix)].to_numpy().tolist()
    return df.drop(columns=remove_columns)


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
                    default='wide',
                    type=str,
                    choices=['half', 'wide'],
                    help='The dimentions and format of the plot.')
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
parser.add_argument('--model',
                    action='store',
                    default='roberta-sb',
                    type=str,
                    help='Which model to use.')
parser.add_argument('--max-masking-ratio',
                    action='store',
                    default=100,
                    type=int,
                    help='The maximum masking ratio (percentage integer) to apply on the training dataset')
parser.add_argument('--masking-strategy',
                    default='half-det',
                    choices=['uni', 'half-det', 'half-ran'],
                    type=str,
                    help='The masking strategy to use for masking during fune-tuning')
parser.add_argument('--validation-dataset',
                    default='both',
                    choices=['nomask', 'mask', 'both'],
                    type=str,
                    help='The transformation applied to the validation dataset used for early stopping.')

if __name__ == "__main__":
    args, unknown = parser.parse_known_args()

    dataset_mapping = pd.DataFrame([
        {
            'args.dataset': dataset_name,
            'target_metric': datasets[dataset_name]._early_stopping_metric if args.performance_metric == 'primary' else args.performance_metric
        }
        for dataset_name in args.datasets
    ])

    experiment_id = generate_experiment_id('epoch_by_valid',
                                           model=args.model,
                                           max_masking_ratio=args.max_masking_ratio,
                                           validation_dataset=args.validation_dataset)

    if args.stage in ['both', 'preprocess']:
        # Read JSON files into dataframe
        results = []
        files = sorted((args.persistent_dir / 'results' / 'masking').glob('*.json'))
        for file in tqdm(files, desc='Loading masking .json files'):
            with open(file, 'r') as fp:
                try:
                    data = json.load(fp)
                except json.decoder.JSONDecodeError:
                    print(f'{file} has a format error')

                if data['args']['max_masking_ratio'] == args.max_masking_ratio and \
                   data['args']['model'] == args.model and \
                   data['args']['dataset'] in args.datasets and \
                   data['args']['validation_dataset'] in args.validation_dataset:
                    results.append(data)

        df = pd.json_normalize(results).explode('history', ignore_index=True)
        results = pd.json_normalize(df.pop('history')).add_prefix('history.')
        df = pd.concat([df, results], axis=1)

        # Select test metric
        args_columns = df.columns[df.columns.str.startswith('args.')].to_numpy().tolist()
        df = (df
              .merge(dataset_mapping, on='args.dataset')
              .transform(partial(select_target_metric, selectors={
                  'metric.train': 'history.',
                  'metric.val': 'history.val_',
                  'metric.val_0': 'history.val_0_',
                  'metric.val_10': 'history.val_10_',
                  'metric.val_20': 'history.val_20_',
                  'metric.val_30': 'history.val_30_',
                  'metric.val_40': 'history.val_40_',
                  'metric.val_50': 'history.val_50_',
                  'metric.val_60': 'history.val_60_',
                  'metric.val_70': 'history.val_70_',
                  'metric.val_80': 'history.val_80_',
                  'metric.val_90': 'history.val_90_',
                  'metric.val_100': 'history.val_100_'
              }))
              .assign(**{'epoch': lambda df: df['history.epoch'] + 1})
              .transform(partial(delete_columns, prefix='history.'))
              .transform(partial(delete_columns, prefix='durations.'))
              .drop(columns=['results', 'target_metric'])
              .melt(id_vars=args_columns + ['epoch'],
                    value_vars=[
                        'metric.val', 'metric.val_0', 'metric.val_10',
                        'metric.val_20', 'metric.val_30', 'metric.val_40',
                        'metric.val_50', 'metric.val_60', 'metric.val_70',
                        'metric.val_80', 'metric.val_90', 'metric.val_100'
              ],
                  value_name='metric.value',
                  var_name='metric.dataset'))

    if args.stage in ['preprocess']:
        os.makedirs(args.persistent_dir / 'pandas', exist_ok=True)
        df.to_parquet((args.persistent_dir / 'pandas' / experiment_id).with_suffix('.parquet'))
    elif args.stage in ['plot']:
        df = pd.read_parquet((args.persistent_dir / 'pandas' / experiment_id).with_suffix('.parquet'))

    if args.stage in ['both', 'plot']:
        df_epochs = (df
                     .groupby(['args.dataset', 'args.masking_strategy', 'epoch', 'metric.dataset'], group_keys=True)
                     .apply(bootstrap_confint(['metric.value']))
                     .reset_index())

        # Generate plot
        p = (p9.ggplot(df_epochs, p9.aes(x='epoch'))
             + p9.geom_ribbon(p9.aes(ymin='metric.value_lower', ymax='metric.value_upper', fill='metric.dataset'), alpha=0.35)
             + p9.geom_line(p9.aes(y='metric.value_mean', color='metric.dataset'))
             + p9.facet_grid("args.masking_strategy ~ args.dataset", scales="free_x")
             + p9.scale_x_continuous(name='Epoch')
             + p9.scale_y_continuous(
            labels=lambda ticks: [f'{tick:.0%}' for tick in ticks],
            name='Performance'
        )
            + p9.scale_color_manual(
                values=['#000000'] + brewer_pal(type='div', palette=8)(11),
                breaks=annotation.validation.breaks,
                labels=annotation.validation.labels,
                aesthetics=["colour", "fill"],
                name='Validation dataset',
        )
            + p9.guides(shape=False))

        if args.format == 'half':
            # The width is the \linewidth of a collumn in the LaTeX document
            size = (3.03209, 4.5)
            p += p9.guides(color=p9.guide_legend(ncol=2))
            p += p9.theme(text=p9.element_text(size=11), subplots_adjust={'bottom': 0.25}, legend_position=(.5, .05))
        else:
            size = (20, 7)
            p += p9.ggtitle(experiment_id)

        os.makedirs(args.persistent_dir / 'plots' / args.format, exist_ok=True)
        p.save(args.persistent_dir / 'plots' / args.format / f'{experiment_id}.pdf', width=size[0], height=size[1], units='in')
        p.save(args.persistent_dir / 'plots' / args.format / f'{experiment_id}.png', width=size[0], height=size[1], units='in')
