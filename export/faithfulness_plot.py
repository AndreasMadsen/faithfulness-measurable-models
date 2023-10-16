
import json
import argparse
import os
import pathlib

from tqdm import tqdm
import pandas as pd
import plotnine as p9
import numpy as np

from ecoroar.dataset import datasets
from ecoroar.plot import bootstrap_confint, annotation
from ecoroar.util import generate_experiment_id
from ecoroar.explain import explainers


def select_target_metric(df):
    idx, cols = pd.factorize('results.' + df.loc[:, 'target_metric'])
    return df.assign(
        metric=df.reindex(cols, axis=1).to_numpy()[np.arange(len(df)), idx]
    )


def check_converged(df):
    unmasked_performance = df.query('`results.masking_ratio` == 0')
    return unmasked_performance['metric'] > unmasked_performance['convergence_threshold']


def annotate_explainer(df):
    sign_lookup = {
        name: 'sign' if explainer._signed else 'abs'
        for name, explainer in explainers.items()
    }
    base_lookup = {
        name: explainer._base_name
        for name, explainer in explainers.items()
    }

    df_annotated = df.assign(**{
        'plot.explainer_sign': df['args.explainer'].map(sign_lookup),
        'plot.explainer_base': df['args.explainer'].map(base_lookup)
    })

    x = pd.concat([
        df_annotated,
        df_annotated.query('`args.explainer` == "rand"').assign(**{
            'plot.explainer_sign': 'sign'
        })
    ])

    return x


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
parser.add_argument('--split',
                    default='test',
                    choices=['train', 'valid', 'test'],
                    type=str,
                    help='The dataset split to evaluate faithfulness on')

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    args, unknown = parser.parse_known_args()

    dataset_mapping = pd.DataFrame([
        {
            'args.dataset': dataset._name,
            'target_metric': dataset._early_stopping_metric if args.performance_metric == 'primary' else args.performance_metric,
            'convergence_threshold': dataset._convergence_threshold,
        }
        for dataset in datasets.values()
    ])

    experiment_id = generate_experiment_id('faithfulness',
                                           model=args.model,
                                           dataset=args.page,
                                           max_masking_ratio=args.max_masking_ratio,
                                           masking_strategy=args.masking_strategy,
                                           split=args.split)

    if args.stage in ['both', 'preprocess']:
        # Read JSON files into dataframe
        results = []
        files = sorted((args.persistent_dir / 'results' / 'faithfulness').glob('faithfulness_*.json'))
        for file in tqdm(files, desc='Loading faithfulness .json files'):
            with open(file, 'r') as fp:
                try:
                    data = json.load(fp)
                except json.decoder.JSONDecodeError:
                    print(f'{file} has a format error')

                if data['args']['max_masking_ratio'] == args.max_masking_ratio and \
                   data['args']['masking_strategy'] == args.masking_strategy and \
                   data['args']['split'] == args.split and \
                   data['args']['model'] in args.model and \
                   data['args']['dataset'] in args.datasets:
                    results.append(data)

        df_faithfulness = pd.json_normalize(results).explode('results', ignore_index=True)
        results = pd.json_normalize(df_faithfulness.pop('results')).add_prefix('results.')
        df_faithfulness = pd.concat([df_faithfulness, results], axis=1)

        # Select test metric
        df = (df_faithfulness
              .merge(dataset_mapping, on='args.dataset')
              .transform(select_target_metric))

    if args.stage in ['preprocess']:
        os.makedirs(args.persistent_dir / 'pandas', exist_ok=True)
        df.to_parquet((args.persistent_dir / 'pandas' / experiment_id).with_suffix('.parquet'))
    elif args.stage in ['plot']:
        df = pd.read_parquet((args.persistent_dir / 'pandas' / experiment_id).with_suffix('.parquet'))

    if args.stage in ['both', 'plot']:
        df_plot = (df
                   .groupby(['args.seed', 'args.dataset', 'args.explainer'], group_keys=True)
                   .filter(check_converged)
                   .reset_index()
                   .groupby(['args.dataset', 'args.explainer', 'results.masking_ratio'], group_keys=True)
                   .apply(bootstrap_confint(['metric']))
                   .reset_index())
        df_plot = annotate_explainer(df_plot)

        # Generate plot
        p = (p9.ggplot(df_plot, p9.aes(x='results.masking_ratio'))
             + p9.geom_ribbon(p9.aes(ymin='metric_lower', ymax='metric_upper', fill='plot.explainer_base'), alpha=0.35)
             + p9.geom_point(p9.aes(y='metric_mean', color='plot.explainer_base'))
             + p9.geom_line(p9.aes(y='metric_mean', color='plot.explainer_base'))
             + p9.facet_grid("args.dataset ~ plot.explainer_sign", scales="free_y",
                             labeller=(annotation.dataset | annotation.explainer_sign).labeller)
             + p9.scale_x_continuous(
            labels=lambda ticks: [f'{tick:.0%}' for tick in ticks],
            name='Masking ratio')
            + p9.scale_y_continuous(
            labels=lambda ticks: [f'{tick:.0%}' for tick in ticks],
            name='IM masked performance'
        )
            + p9.scale_color_discrete(
                breaks=annotation.explainer_base.breaks,
                labels=annotation.explainer_base.labels,
                aesthetics=["colour", "fill"],
                name='importance measure (IM)'
        )
            + p9.scale_shape_discrete(guide=False))

        if args.format == 'half':
            # The width is the \linewidth of a collumn in the LaTeX document
            size = (3.03209, 4.5)
            p += p9.guides(color=p9.guide_legend(ncol=1))
            p += p9.theme(text=p9.element_text(size=11), subplots_adjust={'bottom': 0.38}, legend_position=(.5, .05))
        elif args.format == 'paper':
            # The width is the \linewidth of a collumn in the LaTeX document
            size = (3.03209, 4.4)
            p += p9.guides(color=p9.guide_legend(ncol=3))
            p += p9.scale_y_continuous(
                labels=lambda ticks: [f'{tick:.0%}' for tick in ticks],
                name=f'                      IM masked performance'
            )
            p += p9.theme(
                text=p9.element_text(size=10, fontname='Times New Roman'),
                subplots_adjust={'bottom': 0.30},
                panel_spacing=.05,
                legend_box_margin=0,
                legend_position=(.5, .05),
                legend_background=p9.element_rect(fill='#F2F2F2'),
                strip_background_x=p9.element_rect(height=0.25),
                strip_background_y=p9.element_rect(width=0.2),
                strip_text_x=p9.element_text(margin={'b': 5}),
                axis_text_x=p9.element_text(angle=60, hjust=1)
            )
        elif args.format == 'keynote':
            # The width is the \linewidth of a collumn in the LaTeX document
            size = (3.03209, 4.1)
            p += p9.guides(color=p9.guide_legend(ncol=1))
            p += p9.scale_y_continuous(
                labels=lambda ticks: [f'{tick:.0%}' for tick in ticks],
                name=f'                      IM masked performance'
            )
            p += p9.theme(
                text=p9.element_text(size=10, fontname='Times New Roman'),
                subplots_adjust={'bottom': 0.31},
                panel_spacing=.05,
                legend_box_margin=0,
                legend_position='right',
                legend_background=p9.element_rect(fill='#F2F2F2'),
                strip_background_x=p9.element_rect(height=0.17),
                strip_background_y=p9.element_rect(width=0.2),
                strip_text_x=p9.element_text(margin={'b': 3}),
                axis_text_x=p9.element_text(angle=60, hjust=1)
            )
        elif args.format == 'appendix':
            size = (6.30045, 8.5)
            p += p9.guides(color=p9.guide_legend(ncol=5))
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
            size = (20, 7)
            p += p9.ggtitle(experiment_id)

        os.makedirs(args.persistent_dir / 'plots' / args.format, exist_ok=True)
        p.save(args.persistent_dir / 'plots' / args.format / f'{experiment_id}.pdf', width=size[0], height=size[1], units='in')
        # p.save(args.persistent_dir / 'plots'/ args.format / f'{experiment_id}.png', width=size[0], height=size[1], units='in')
