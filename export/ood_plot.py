
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
                    choices=['half', 'wide', 'paper', 'appendix'],
                    help='The dimentions and format of the plot.')
parser.add_argument('--datasets',
                    action='store',
                    nargs='+',
                    default=list(datasets.keys()),
                    choices=datasets.keys(),
                    type=str,
                    help='The datasets to plot')
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
parser.add_argument('--split',
                    default='test',
                    choices=['train', 'valid', 'test'],
                    type=str,
                    help='The dataset split to evaluate faithfulness on')
parser.add_argument('--ood',
                    default='MaSF',
                    choices=['MaSF'],
                    type=str,
                    help='The OOD detection method')
parser.add_argument('--dist-repeats',
                    default=2,
                    type=int,
                    help='The number of repeats used to estimate the distribution')
parser.add_argument('--threshold',
                    default=0.05,
                    choices=[0.001, 0.005, 0.01, 0.05, 0.1],
                    type=float,
                    help='The p-value threshold')

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    args, unknown = parser.parse_known_args()

    model_categories = {
        'masking-ratio': ['roberta-m15', 'roberta-m20', 'roberta-m30', 'roberta-m40', 'roberta-m50'],
        'size': ['roberta-sb', 'roberta-sl']
    }

    experiment_id = generate_experiment_id(
        'ood',
        model=args.model_category,
        max_masking_ratio=args.max_masking_ratio,
        masking_strategy=args.masking_strategy,
        validation_dataset=args.validation_dataset,
        split=args.split,
        ood=args.ood,
        dist_repeats=args.dist_repeats
    )

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

                if data['args']['max_masking_ratio'] == args.max_masking_ratio and \
                   data['args']['masking_strategy'] == args.masking_strategy and \
                   data['args']['validation_dataset'] == args.validation_dataset and \
                   data['args']['split'] == args.split and \
                   data['args']['ood'] == args.ood and \
                   data['args']['dist_repeats'] == args.dist_repeats and \
                   data['args']['model'] in model_categories[args.model_category] and \
                   data['args']['dataset'] in args.datasets:
                    results.append(data)

        df = pd.json_normalize(results).explode('results', ignore_index=True)
        results = pd.json_normalize(df.pop('results')).add_prefix('results.')
        df = pd.concat([df, results], axis=1)

    if args.stage in ['preprocess']:
        os.makedirs(args.persistent_dir / 'pandas', exist_ok=True)
        df.to_parquet((args.persistent_dir / 'pandas' / experiment_id).with_suffix('.parquet'))
    elif args.stage in ['plot']:
        df = pd.read_parquet((args.persistent_dir / 'pandas' / experiment_id).with_suffix('.parquet'))

    if args.stage in ['both', 'plot']:
        df_plot = (df
            .query('`results.threshold` == @args.threshold')
            .groupby(['args.model', 'args.dataset', 'args.explainer',
                      'results.masking_ratio'], group_keys=True)
            .apply(bootstrap_confint(['results.proportion']))
            .reset_index())

        df_baseline = (df
            .groupby(['args.model', 'args.dataset', 'results.masking_ratio'], group_keys=True)
            .apply(lambda _: pd.Series({ 'threshold': args.threshold }))
            .reset_index())

        # Generate plot
        p = (p9.ggplot(df_plot, p9.aes(x='results.masking_ratio'))
            + p9.geom_ribbon(p9.aes(ymin='results.proportion_lower', ymax='results.proportion_upper', fill='args.explainer'), alpha=0.35)
            + p9.geom_point(p9.aes(y='results.proportion_mean', color='args.explainer'))
            + p9.geom_line(p9.aes(y='results.proportion_mean', color='args.explainer'))
            + p9.geom_line(p9.aes(y='threshold'), color='black', data=df_baseline)
            + p9.facet_grid("args.dataset ~ args.model", scales="free_y", labeller=annotation.model.labeller)
            + p9.scale_x_continuous(
                labels=lambda ticks: [f'{tick:.0%}' for tick in ticks],
                name='Masking ratio')
            + p9.scale_y_continuous(
                labels=lambda ticks: [f'{tick:.0%}' for tick in ticks],
                limits=[0, None],
                name=f'p-value < {args.threshold:.0%}'
            )
            + p9.scale_color_discrete(
                breaks = annotation.explainer.breaks,
                labels = annotation.explainer.labels,
                aesthetics = ["colour", "fill"],
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
            size = (3.03209, 4.5)
            p += p9.guides(color=p9.guide_legend(ncol=3))
            p += p9.scale_y_continuous(
                labels=lambda ticks: [f'{tick:.0%}' for tick in ticks],
                limits=[0, None],
                name=f'              p-value < {args.threshold:.0%}'
            )
            p += p9.theme(
                text=p9.element_text(size=11, fontname='Times New Roman'),
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
            size = (6.30045, 9)
            p += p9.guides(color=p9.guide_legend(ncol=4))
            p += p9.theme(
                text=p9.element_text(size=11, fontname='Times New Roman'),
                subplots_adjust={'bottom': 0.18},
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
        p.save(args.persistent_dir / 'plots'/ args.format / f'{experiment_id}.png', width=size[0], height=size[1], units='in')
