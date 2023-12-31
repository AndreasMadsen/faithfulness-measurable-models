
import json
import argparse
import os
import pathlib

from tqdm import tqdm
import numpy as np
import pandas as pd
import plotnine as p9

from ecoroar.dataset import datasets
from ecoroar.plot import bootstrap_confint, annotation
from ecoroar.util import generate_experiment_id


def select_target_metric(df):
    idx, cols = pd.factorize('results.' + df.loc[:, 'target_metric'])
    return df.assign(
        metric=df.reindex(cols, axis=1).to_numpy()[np.arange(len(df)), idx]
    )


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
parser.add_argument('--model-category',
                    action='store',
                    default='size',
                    type=str,
                    choices=['size', 'masking-ratio'],
                    help='Which model category to use.')
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
    pd.set_option('display.max_rows', None)
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

    experiment_id = generate_experiment_id('unmasked_performance_by_mmr',
                                           model=args.model_category,
                                           masking_strategy=args.masking_strategy,
                                           validation_dataset=args.validation_dataset)

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

                if data['args']['masking_strategy'] == args.masking_strategy and \
                   data['args']['model'] in model_categories[args.model_category] and \
                   data['args']['dataset'] in args.datasets and \
                   data['args']['validation_dataset'] == args.validation_dataset:
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
        df.to_parquet((args.persistent_dir / 'pandas' / experiment_id).with_suffix('.parquet'))
    elif args.stage in ['plot']:
        df = pd.read_parquet((args.persistent_dir / 'pandas' / experiment_id).with_suffix('.parquet'))

    if args.stage in ['both', 'plot']:
        df_plot = (df
                   .groupby(['args.model', 'args.dataset', 'args.max_epochs', 'args.max_masking_ratio'])
                   .apply(bootstrap_confint(['metric']))
                   .reset_index()
                   )

        df_goal = df_plot.query('`args.max_masking_ratio` == 0')
        df_goal = pd.concat([
            df_goal.assign(**{'args.max_masking_ratio': max_masking_ratio})
            for max_masking_ratio in [0, 100]
        ])

        # Generate plot
        p = (p9.ggplot(df_plot, p9.aes(x='args.max_masking_ratio'))
             + p9.geom_jitter(p9.aes(y='metric', group='args.seed', color='args.model'),
                              shape='+', alpha=0.5, width=1, data=df)
             + p9.geom_ribbon(p9.aes(ymin='metric_lower', ymax='metric_upper', fill='args.model'), alpha=0.35)
             + p9.geom_line(p9.aes(y='metric_mean', color='args.model'))
             + p9.geom_point(p9.aes(y='metric_mean', color='args.model', shape='args.model'))
             + p9.geom_line(p9.aes(y='metric_mean', color='args.model'), linetype='dashed', data=df_goal)
             + p9.facet_wrap("args.dataset", scales="free_y", ncol=2)
             + p9.scale_y_continuous(
            labels=lambda ticks: [f'{tick:.0%}' for tick in ticks],
            name='Unmasked performance'
        )
            + p9.scale_x_continuous(
                labels=lambda ticks: [f'{tick:.0f}%' for tick in ticks],
                name='Max masking ratio'
        )
            + p9.scale_color_discrete(
                breaks=annotation.model.breaks,
                labels=annotation.model.labels,
                aesthetics=["colour", "fill"],
                name='Model'
        )
            + p9.scale_shape_discrete(guide=False)
            + p9.theme(subplots_adjust={'wspace': 0.25}))

        if args.format == 'half':
            # The width is the \linewidth of a collumn in the LaTeX document
            size = (3.03209, 4.5)
            p += p9.guides(color=p9.guide_legend(ncol=2))
            p += p9.theme(
                text=p9.element_text(size=11),
                subplots_adjust={'bottom': 0.37, 'wspace': 0.5},
                legend_position=(.5, .05),
                axis_text_x=p9.element_text(angle=45, hjust=1)
            )
        else:
            size = (20, 7)
            p += p9.ggtitle(experiment_id)

        os.makedirs(args.persistent_dir / 'plots' / args.format, exist_ok=True)
        p.save(args.persistent_dir / 'plots' / args.format / f'{experiment_id}.pdf', width=size[0], height=size[1], units='in')
        p.save(args.persistent_dir / 'plots' / args.format / f'{experiment_id}.png', width=size[0], height=size[1], units='in')
