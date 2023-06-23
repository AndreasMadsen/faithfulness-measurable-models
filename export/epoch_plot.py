
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
                    default='wide',
                    type=str,
                    choices=['half', 'full', 'paper', 'appendix'],
                    help='The dimentions and format of the plot.')
parser.add_argument('--page',
                    action='store',
                    default=None,
                    type=str,
                    help='The page name')
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
parser.add_argument('--model-category',
                    action='store',
                    default='size',
                    type=str,
                    choices=['size', 'masking-ratio'],
                    help='Which model category to use.')
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
    #pd.set_option('display.max_rows', None)
    args, unknown = parser.parse_known_args()

    dataset_mapping = pd.DataFrame([
        {
            'args.dataset': dataset_name,
            'target_metric': datasets[dataset_name]._early_stopping_metric if args.performance_metric == 'primary' else args.performance_metric
        }
        for dataset_name in args.datasets
    ])
    model_categories = {
        'masking-ratio': ['roberta-m15', 'roberta-m20', 'roberta-m30', 'roberta-m40', 'roberta-m50'],
        'size': ['roberta-sb', 'roberta-sl']
    }

    experiment_id = generate_experiment_id('epoch',
                                            model=args.model_category,
                                            dataset=args.page,
                                            masking_strategy=args.masking_strategy,
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

                if data['args']['masking_strategy'] == args.masking_strategy and \
                   data['args']['max_masking_ratio'] in [0, args.max_masking_ratio] and \
                   data['args']['model'] in model_categories[args.model_category] and \
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
                'metric.val_0': 'history.val_0_',
              }))
              .assign(**{'epoch': lambda df: df['history.epoch'] + 1})
              .transform(partial(delete_columns, prefix='history.'))
              .transform(partial(delete_columns, prefix='durations.'))
              .drop(columns=['results', 'target_metric']))

    if args.stage in ['preprocess']:
        os.makedirs(args.persistent_dir / 'pandas', exist_ok=True)
        df.to_parquet((args.persistent_dir / 'pandas' / experiment_id).with_suffix('.parquet'))
    elif args.stage in ['plot']:
        df = pd.read_parquet((args.persistent_dir / 'pandas' / experiment_id).with_suffix('.parquet'))

    if args.stage in ['both', 'plot']:
        df = df.assign(**{
            'plot.max_masking_ratio': df['args.max_masking_ratio'].astype(str)
        })

        df_epochs = (df
            .groupby(['args.model', 'args.dataset', 'plot.max_masking_ratio', 'epoch'], group_keys=True)
            .apply(bootstrap_confint(['metric.val_0']))
            .reset_index())

        # Generate plot
        p = (p9.ggplot(df_epochs, p9.aes(x='epoch'))
            + p9.geom_ribbon(p9.aes(ymin='metric.val_0_lower', ymax='metric.val_0_upper', fill='plot.max_masking_ratio'), alpha=0.35)
            + p9.geom_line(p9.aes(y='metric.val_0_mean', color='plot.max_masking_ratio'))
            + p9.geom_jitter(p9.aes(y='metric.val_0', color='plot.max_masking_ratio'),
                             shape='+', alpha=0.5, position=p9.position_jitterdodge(0.05), data=df)
            + p9.facet_grid("args.dataset ~ args.model", scales="free_y", labeller=annotation.model.labeller)
            + p9.scale_x_continuous(name='Epoch')
            + p9.scale_y_continuous(
                labels=lambda ticks: [f'{tick:.0%}' for tick in ticks],
                name='0% masked validation performance'
            )
            + p9.scale_color_discrete(
                breaks = annotation.max_masking_ratio.breaks,
                labels = annotation.max_masking_ratio.labels,
                aesthetics = ["colour", "fill"],
                name='Fine-tuning method'
            )
            + p9.scale_shape_discrete(guide=False))

        if args.format == 'half':
            # The width is the \linewidth of a collumn in the LaTeX document
            size = (3.03209, 4.5)
            p += p9.guides(color=p9.guide_legend(ncol=2))
            p += p9.theme(text=p9.element_text(size=11), subplots_adjust={'bottom': 0.25}, legend_position=(.5, .05))
        elif args.format == 'paper':
            # The width is the \linewidth of a collumn in the LaTeX document
            size = (3.03209, 4.5)
            p += p9.guides(color=p9.guide_legend(ncol=3))
            p += p9.theme(
                text=p9.element_text(size=10, fontname='Times New Roman'),
                subplots_adjust={'bottom': 0.31},
                panel_spacing=.05,
                legend_box_margin=0,
                legend_position=(.5, .05),
                legend_background=p9.element_rect(fill='#F2F2F2'),
                strip_background_x=p9.element_rect(height=0.2),
                strip_background_y=p9.element_rect(width=0.2),
                strip_text_x=p9.element_text(margin={'b': 5}),
                axis_text_x=p9.element_text(angle = 60, hjust=1)
            )
        elif args.format == 'appendix':
            size = (6.30045, 8.6)
            p += p9.guides(color=p9.guide_legend(ncol=4))
            p += p9.theme(
                text=p9.element_text(size=10, fontname='Times New Roman'),
                subplots_adjust={'bottom': 0.14},
                panel_spacing=.05,
                legend_box_margin=0,
                legend_position=(.5, .05),
                legend_background=p9.element_rect(fill='#F2F2F2'),
                axis_text_x=p9.element_text(angle = 15, hjust=1)
            )
        else:
            size = (20, 7)
            p += p9.ggtitle(experiment_id)

        os.makedirs(args.persistent_dir / 'plots' / args.format, exist_ok=True)
        p.save(args.persistent_dir / 'plots'/ args.format / f'{experiment_id}.pdf', width=size[0], height=size[1], units='in')
        # p.save(args.persistent_dir / 'plots'/ args.format / f'{experiment_id}.png', width=size[0], height=size[1], units='in')
