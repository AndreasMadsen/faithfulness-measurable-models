
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
                    choices=['half', 'full', 'paper', 'appendix'],
                    help='The dimentions and format of the plot.')
parser.add_argument('--datasets',
                    action='store',
                    nargs='+',
                    default=list(datasets.keys()),
                    choices=datasets.keys(),
                    type=str,
                    help='The datasets to plot')
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
parser.add_argument('--split',
                    default='test',
                    choices=['train', 'valid', 'test'],
                    type=str,
                    help='The dataset split to evaluate faithfulness on')
parser.add_argument('--ood',
                    default='masf',
                    choices=['masf', 'masf-slow'],
                    type=str,
                    help='The OOD detection method')
parser.add_argument('--dist-repeats',
                    default=2,
                    type=int,
                    help='The number of repeats used to estimate the distribution')
parser.add_argument('--method',
                    default='proportion',
                    choices=['proportion', 'simes', 'fisher'],
                    type=str,
                    help='The p-value aggregation method')
parser.add_argument('--threshold',
                    default=0.05,
                    choices=[0.001, 0.005, 0.01, 0.05, 0.1],
                    type=float,
                    help='The p-value threshold, relevant only for the value method')
parser.add_argument('--model-baseline',
                    action=argparse.BooleanOptionalAction,
                    default=False,
                    help='Should a baseline for the unmasked model be included')

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    args, unknown = parser.parse_known_args()

    experiment_id = generate_experiment_id(
        f'ood_a-{args.method}_p-{args.threshold}',
        model=args.model,
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

                if data['args']['max_masking_ratio'] in [0, args.max_masking_ratio] and \
                   data['args']['masking_strategy'] == args.masking_strategy and \
                   data['args']['validation_dataset'] == args.validation_dataset and \
                   data['args']['split'] == args.split and \
                   data['args']['ood'] == args.ood and \
                   data['args']['dist_repeats'] == args.dist_repeats and \
                   data['args']['model'] == args.model and \
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
        # select method and threshold
        df_data = df.query('`results.method` == @args.method')
        if args.method == 'proportion':
            df_data = df.query('`results.threshold` == @args.threshold')

        df_plot = (df_data
            .groupby(['args.dataset', 'args.max_masking_ratio', 'args.explainer',
                      'results.masking_ratio'], group_keys=True)
            .apply(bootstrap_confint(['results.value']))
            .reset_index())

        # Create model baseline
        df_baseline_model = (df_data
            .query('`args.max_masking_ratio` == 0 & `args.explainer` == "rand" & `results.masking_ratio` == 0')
            .groupby(['args.dataset'], group_keys=True)
            .apply(bootstrap_confint(['results.value']))
            .reset_index())
        df_baseline_model = pd.concat([
            df_baseline_model.assign(**{
                'args.max_masking_ratio': max_masking_ratio,
            })
            for max_masking_ratio in [0, args.max_masking_ratio]
        ])
        df_baseline_model_with_x_axis = pd.concat([
            df_baseline_model.assign(**{
                'results.masking_ratio': masking_ratio,
            })
            for masking_ratio in [-1, 2]
        ])
        # Remove content if model_baseline is disabled
        if not args.model_baseline:
            df_baseline_model = df_baseline_model.iloc[:0, :]
            df_baseline_model_with_x_axis = df_baseline_model_with_x_axis.iloc[:0, :]

        # Creat p-value baseline
        df_baseline_pvalue = (df_data
            .groupby(['args.max_masking_ratio', 'args.dataset'], group_keys=True)
            .apply(lambda _: pd.Series({ 'threshold': args.threshold }))
            .reset_index())

        # Conditional y-axis name
        if args.method == 'proportion':
            y_axis_name = f'p-value < {args.threshold:.0%}'
        else:
            y_axis_name = 'p-value'

        # Generate plot
        p = (p9.ggplot(df_plot, p9.aes(x='results.masking_ratio'))
            + p9.geom_hline(p9.aes(yintercept='threshold'), color='black', linetype='dashed', data=df_baseline_pvalue)
            + p9.geom_ribbon(p9.aes(ymin='results.value_lower', ymax='results.value_upper'), fill='green', alpha=0.35, data=df_baseline_model_with_x_axis)
            + p9.geom_hline(p9.aes(yintercept='results.value_mean'), color='green', linetype='dashed', data=df_baseline_model)
            + p9.geom_ribbon(p9.aes(ymin='results.value_lower', ymax='results.value_upper', fill='args.explainer'), alpha=0.35)
            + p9.geom_point(p9.aes(y='results.value_mean', color='args.explainer'))
            + p9.geom_line(p9.aes(y='results.value_mean', color='args.explainer'))
            + p9.facet_grid("args.dataset ~ args.max_masking_ratio", scales="free_y", labeller=(annotation.dataset | annotation.max_masking_ratio).labeller)
            + p9.scale_x_continuous(
                labels=lambda ticks: [f'{tick:.0%}' for tick in ticks],
                name='Masking ratio')
            + p9.coord_cartesian(xlim=[0, 1])
            + p9.scale_y_continuous(
                labels=lambda ticks: [f'{tick:.0%}' for tick in ticks],
                limits=[0, None],
                name=y_axis_name
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
            size = (3.03209, 3.5)
            p += p9.guides(color=p9.guide_legend(ncol=3))
            p += p9.scale_y_continuous(
                labels=lambda ticks: [f'{tick:.0%}' for tick in ticks],
                limits=[0, None],
                name=f'                             {y_axis_name}'
            )
            p += p9.theme(
                text=p9.element_text(size=10, fontname='Times New Roman'),
                subplots_adjust={'bottom': 0.41},
                panel_spacing=.05,
                legend_box_margin=0,
                legend_position=(.5, .05),
                legend_background=p9.element_rect(fill='#F2F2F2'),
                strip_background_x=p9.element_rect(height=0.25),
                strip_background_y=p9.element_rect(width=0.2),
                strip_text_x=p9.element_text(margin={'b': 5}),
                axis_text_x=p9.element_text(angle = 60, hjust=1)
            )
        elif args.format == 'appendix':
            size = (6.30045, 8.6)
            p += p9.guides(color=p9.guide_legend(ncol=5))
            p += p9.theme(
                text=p9.element_text(size=10, fontname='Times New Roman'),
                subplots_adjust={'bottom': 0.16},
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
        #p.save(args.persistent_dir / 'plots'/ args.format / f'{experiment_id}.png', width=size[0], height=size[1], units='in')
