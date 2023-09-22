
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


def get_validation_performance(history, test_results):
    losses = [epoch_losses['loss'] for epoch_losses in history]
    best_epoch_losses = history[np.argmin(losses)]

    test_keys = test_results[0].keys()
    val_results = [
        {
            metric_name: (
                masking_ratio
                if metric_name == 'masking_ratio'
                else best_epoch_losses[f'val_{masking_ratio*100:.0f}_{metric_name}']
            ) for metric_name in test_keys
        }
        for masking_ratio in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    ]
    return val_results


parser = argparse.ArgumentParser(
    description='Plots the 0% masking test performance given different training masking ratios')
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
                    choices=['half', 'wide', 'paper', 'keynote', 'appendix'],
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
parser.add_argument('--aggregate',
                    action='store',
                    nargs='*',
                    default=[],
                    choices=datasets.keys(),
                    type=str,
                    help='Datasets the macro-average should be calculated over')
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
parser.add_argument('--split',
                    action='store',
                    default='test',
                    type=str,
                    choices=['test', 'valid'],
                    help='Either test or valid, chooses which dataset to take results from')

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    args, unknown = parser.parse_known_args()

    all_datasets = set(args.datasets + args.aggregate)

    dataset_mapping = pd.DataFrame([
        {
            'args.dataset': dataset._name,
            'target_metric': dataset._early_stopping_metric if args.performance_metric == 'primary' else args.performance_metric,
            'baseline': dataset.majority_classifier_performance(args.split)[
                dataset._early_stopping_metric if args.performance_metric == 'primary' else args.performance_metric
            ]
        }
        for dataset in datasets.values()
    ])
    model_categories = {
        'masking-ratio': ['roberta-m15', 'roberta-m20', 'roberta-m30', 'roberta-m40', 'roberta-m50'],
        'size': ['roberta-sb', 'roberta-sl']
    }

    experiment_id = generate_experiment_id('masked_100p_performance_by_valid_ms',
                                           model=args.model_category,
                                           dataset=args.page,
                                           max_masking_ratio=args.max_masking_ratio,
                                           split=args.split)

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

                if data['args']['max_masking_ratio'] in [0, args.max_masking_ratio] and \
                   data['args']['model'] in model_categories[args.model_category] and \
                   data['args']['dataset'] in all_datasets:
                    if args.split == 'valid':
                        data['results'] = get_validation_performance(data['history'], data['results'])
                    results.append(data)

        df = pd.json_normalize(results).explode('results', ignore_index=True)
        results = pd.json_normalize(df.pop('results')).add_prefix('results.')
        df = pd.concat([df, results], axis=1)

        # Select test metric
        df = (df
              .merge(dataset_mapping, on='args.dataset')
              .transform(select_target_metric)
              .query('`results.masking_ratio` == 1'))

    if args.stage in ['preprocess']:
        os.makedirs(args.persistent_dir / 'pandas', exist_ok=True)
        df.to_parquet((args.persistent_dir / 'pandas' / experiment_id).with_suffix('.parquet'))
    elif args.stage in ['plot']:
        df = pd.read_parquet((args.persistent_dir / 'pandas' / experiment_id).with_suffix('.parquet'))

    if args.stage in ['both', 'plot']:
        df_main = df.query('`args.max_masking_ratio` == 100')
        df_goal = (df
                   .query('`args.max_masking_ratio` == 0')
                   .assign(**{
                       'args.masking_strategy': 'goal'
                   }))
        df_all = pd.concat([df_main, df_goal])
        df_show = df_all.query(' | '.join(f'`args.dataset` == "{dataset}"' for dataset in args.datasets))

        df_plot = (df_show
                   .groupby(['args.model', 'args.dataset', 'args.max_epochs', 'args.validation_dataset', 'args.masking_strategy'], group_keys=True)
                   .apply(bootstrap_confint(['metric', 'baseline']))
                   .reset_index())

        if len(args.aggregate) > 0:
            df_agg = (df_all
                      .query(' | '.join(f'`args.dataset` == "{dataset}"' for dataset in args.aggregate))
                      .groupby(['args.seed', 'args.model', 'args.validation_dataset', 'args.masking_strategy'], group_keys=True)
                      .apply(lambda subset: pd.Series({
                          'metric': subset['metric'].mean(),
                          'baseline': subset['baseline'].mean()
                      }))
                      .groupby(['args.model', 'args.validation_dataset', 'args.masking_strategy'], group_keys=True)
                      .apply(bootstrap_confint(['metric', 'baseline']))
                      .reset_index()
                      .assign(**{'args.dataset': 'All'}))
            df_plot = pd.concat([df_agg, df_plot])

        df_baseline = (df_plot
                       .query('`args.validation_dataset` == "nomask" & `args.masking_strategy` == "goal"')
                       .drop(columns=['args.validation_dataset', 'args.masking_strategy']))

        # Generate plot
        p = (p9.ggplot(df_plot, p9.aes(x='args.validation_dataset'))
             + p9.geom_hline(p9.aes(yintercept='baseline_mean'), linetype='dashed', data=df_baseline)
             + p9.geom_errorbar(p9.aes(ymin='metric_lower', ymax='metric_upper',
                                color='args.masking_strategy'), position=p9.position_dodge(0.5), width=0.5)
             + p9.geom_point(p9.aes(y='metric_mean', color='args.masking_strategy'),
                             fill='black', shape='o', position=p9.position_dodge(0.5), alpha=1)
             + p9.geom_jitter(p9.aes(y='metric', color='args.masking_strategy'),
                              shape='+', alpha=0.8, position=p9.position_jitterdodge(0.15), data=df_show)
             + p9.facet_grid("args.dataset ~ args.model", scales="free_y",
                             labeller=(annotation.dataset | annotation.model).labeller)
             + p9.scale_y_continuous(
            labels=lambda ticks: [f'{tick:.0%}' for tick in ticks],
            name='100% masked performance'
        )
            + p9.scale_x_discrete(
                breaks=annotation.validation_dataset.breaks,
                labels=annotation.validation_dataset.labels,
                name='Validation strategy'
        )
            + p9.scale_color_discrete(
                 breaks=annotation.masking_strategy.breaks,
                 labels=annotation.masking_strategy.labels,
                 aesthetics=["colour", "fill"],
                 name='Training strategy'
        )
            + p9.scale_shape_discrete(guide=False))

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
        elif args.format == 'paper':
            # The width is the \linewidth of a collumn in the LaTeX document
            size = (3.03209, 4.5)
            p += p9.guides(color=p9.guide_legend(ncol=3))
            p += p9.scale_y_continuous(
                labels=lambda ticks: [f'{tick:.0%}' for tick in ticks],
                name='                              100% masked performance'
            )
            p += p9.theme(
                text=p9.element_text(size=10, fontname='Times New Roman'),
                subplots_adjust={'bottom': 0.31},
                panel_spacing=.05,
                legend_box_margin=0,
                legend_position=(.5, .05),
                legend_background=p9.element_rect(fill='#F2F2F2'),
                strip_background_x=p9.element_rect(height=0.25),
                strip_background_y=p9.element_rect(width=0.2),
                strip_text_x=p9.element_text(margin={'b': 2}),
                axis_text_x=p9.element_text(angle=60, hjust=2)
            )
        elif args.format == 'keynote':
            size = (3.03209, 4.5)
            p += p9.guides(color=p9.guide_legend(ncol=1))
            p += p9.scale_y_continuous(
                labels=lambda ticks: [f'{tick:.0%}' for tick in ticks],
                name='                              100% masked performance'
            )
            p += p9.theme(
                text=p9.element_text(size=10, fontname='Times New Roman'),
                subplots_adjust={'bottom': 0.31},
                panel_spacing=.05,
                legend_box_margin=0,
                legend_position='right',
                legend_background=p9.element_rect(fill='#F2F2F2'),
                strip_background_x=p9.element_rect(height=0.25),
                strip_background_y=p9.element_rect(width=0.2),
                strip_text_x=p9.element_text(margin={'b': 2}),
                axis_text_x=p9.element_text(angle=60, hjust=2)
            )
        elif args.format == 'appendix':
            size = (6.30045, 8.8)
            p += p9.guides(color=p9.guide_legend(ncol=3))
            p += p9.theme(
                text=p9.element_text(size=10, fontname='Times New Roman'),
                subplots_adjust={'bottom': 0.15},
                panel_spacing=.05,
                legend_box_margin=0,
                legend_position=(.5, .05),
                legend_background=p9.element_rect(fill='#F2F2F2'),
                axis_text_x=p9.element_text(angle=15, hjust=1)
            )
        else:
            size = (7, 20)
            p += p9.ggtitle(experiment_id)

        os.makedirs(args.persistent_dir / 'plots' / args.format, exist_ok=True)
        p.save(args.persistent_dir / 'plots' / args.format / f'{experiment_id}.pdf', width=size[0], height=size[1], units='in')
        # p.save(args.persistent_dir / 'plots'/ args.format / f'{experiment_id}.png', width=size[0], height=size[1], units='in')
