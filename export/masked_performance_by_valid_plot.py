
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

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    args, unknown = parser.parse_known_args()

    dataset_mapping = pd.DataFrame([
        {
            'args.dataset': dataset._name,
            'target_metric': dataset._early_stopping_metric if args.performance_metric == 'primary' else args.performance_metric,
            'baseline': dataset.majority_classifier_test_performance()[
                dataset._early_stopping_metric if args.performance_metric == 'primary' else args.performance_metric
            ]
        }
        for dataset in datasets.values()
    ])

    model_categories = {
        'masking-ratio': ['roberta-m15', 'roberta-m20', 'roberta-m30', 'roberta-m40', 'roberta-m50'],
        'size': ['roberta-sb', 'roberta-sl']
    }

    experiment_id = generate_experiment_id('masked_performance_by_valid',
                                            model=args.model_category,
                                            max_masking_ratio=args.max_masking_ratio,
                                            masking_strategy=args.masking_strategy)

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

                if data['args']['max_masking_ratio'] == args.max_masking_ratio and \
                   data['args']['model'] in model_categories[args.model_category] and \
                   data['args']['dataset'] in args.datasets and \
                   data['args']['masking_strategy'] == args.masking_strategy:
                    results.append(data)

        df = pd.json_normalize(results).explode('results', ignore_index=True)
        results = pd.json_normalize(df.pop('results')).add_prefix('results.')
        df = pd.concat([df, results], axis=1)

        # Select test metric
        df = (df
              .merge(dataset_mapping, on='args.dataset')
              .transform(select_target_metric))

    if args.stage in ['preprocess']:
        os.makedirs(args.persistent_dir / 'pandas', exist_ok=True)
        df.to_parquet((args.persistent_dir / 'pandas' / experiment_id).with_suffix('.parquet'))
    elif args.stage in ['plot']:
        df = pd.read_parquet((args.persistent_dir / 'pandas' / experiment_id).with_suffix('.parquet'))

    if args.stage in ['both', 'plot']:
        df_plot = (df
                .groupby(['args.model', 'args.dataset', 'args.validation_dataset',
                          'results.masking_ratio'], group_keys=True)
                .apply(bootstrap_confint(['metric']))
                .reset_index())

        df_baseline = (df
            .groupby(['args.model', 'args.dataset',
                      'results.masking_ratio'], group_keys=True)
            .apply(bootstrap_confint(['baseline']))
            .reset_index())

        # Generate plot
        p = (p9.ggplot(df_plot, p9.aes(x='results.masking_ratio'))
            + p9.geom_jitter(p9.aes(y='metric', group='args.seed', color='args.validation_dataset'),
                            shape='+', alpha=0.5, width=0.01, data=df)
            + p9.geom_ribbon(p9.aes(ymin='metric_lower', ymax='metric_upper', fill='args.validation_dataset'), alpha=0.35)
            + p9.geom_line(p9.aes(y='metric_mean', color='args.validation_dataset', shape='args.model'))
            + p9.geom_point(p9.aes(y='metric_mean', color='args.validation_dataset', shape='args.model'))
            + p9.geom_line(p9.aes(y='baseline_mean'), color='black', data=df_baseline)
            + p9.facet_grid("args.model ~ args.dataset", scales="free_y", labeller=annotation.model.labeller)
            + p9.scale_y_continuous(
                labels=lambda ticks: [f'{tick:.0%}' for tick in ticks],
                name='Masked performance'
            )
            + p9.scale_x_continuous(
                labels=lambda ticks: [f'{tick:.0%}' for tick in ticks],
                name='Test masking ratio'
            )
            + p9.scale_color_discrete(
                breaks = annotation.validation_dataset.breaks,
                labels = annotation.validation_dataset.labels,
                aesthetics = ["colour", "fill"],
                name='Validation dataset'
            )
            + p9.scale_shape_discrete(guide=False))

        if args.format == 'half':
            # The width is the \linewidth of a collumn in the LaTeX document
            size = (3.03209, 6)
            p += p9.guides(color=p9.guide_legend(ncol=2))
            p += p9.theme(text=p9.element_text(size=11), subplots_adjust={'bottom': 0.25}, legend_position=(.5, .05))
        else:
            size = (20, 7)
            p += p9.ggtitle(experiment_id)

        os.makedirs(f'{args.persistent_dir}/plots', exist_ok=True)
        p.save(f'{args.persistent_dir}/plots/{experiment_id}.pdf', width=size[0], height=size[1], units='in')
        p.save(f'{args.persistent_dir}/plots/{experiment_id}.png', width=size[0], height=size[1], units='in')
