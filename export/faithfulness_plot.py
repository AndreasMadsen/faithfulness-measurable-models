
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

    experiment_id = generate_experiment_id('faithfulness',
                                            model=args.model_category,
                                            max_masking_ratio=args.max_masking_ratio,
                                            masking_strategy=args.masking_strategy,
                                            split=args.split)

    if args.stage in ['both', 'preprocess']:
        # Read JSON files into dataframe
        results = []
        files = sorted((args.persistent_dir / 'results').glob('faithfulness_*.json'))
        for file in tqdm(files, desc='Loading faithfulness .json files'):
            with open(file, 'r') as fp:
                try:
                    data = json.load(fp)
                except json.decoder.JSONDecodeError:
                    print(f'{file} has a format error')

                if data['args']['max_masking_ratio'] == args.max_masking_ratio and \
                   data['args']['masking_strategy'] == args.masking_strategy and \
                   data['args']['split'] == args.split and \
                   data['args']['model'] in model_categories[args.model_category]:
                    results.append(data)

        df_faithfulness = pd.json_normalize(results).explode('results', ignore_index=True)
        results = pd.json_normalize(df_faithfulness.pop('results')).add_prefix('results.')
        df_faithfulness = pd.concat([df_faithfulness, results], axis=1)

        # Select test metric
        df = (df_faithfulness
              .merge(dataset_mapping, on='args.dataset')
              .transform(select_target_metric))

    if args.stage in ['preprocess']:
        os.makedirs(f'{args.persistent_dir}/pandas', exist_ok=True)
        df.to_parquet((args.persistent_dir / 'pandas' / experiment_id).with_suffix('.parquet'))
    elif args.stage in ['plot']:
        df = pd.read_parquet((args.persistent_dir / 'pandas' / experiment_id).with_suffix('.parquet'))

    if args.stage in ['both', 'plot']:
        df_plot = (df
            .groupby(['args.model', 'args.dataset', 'args.explainer', 'results.masking_ratio'], group_keys=True)
            .apply(bootstrap_confint(['metric']))
            .reset_index())

        df_baseline = (df
            .groupby(['args.model', 'args.dataset', 'results.masking_ratio'], group_keys=True)
            .apply(bootstrap_confint(['baseline']))
            .reset_index())

        # Generate plot
        p = (p9.ggplot(df_plot, p9.aes(x='results.masking_ratio'))
            + p9.geom_ribbon(p9.aes(ymin='metric_lower', ymax='metric_upper', fill='args.explainer'), alpha=0.35)
            + p9.geom_point(p9.aes(y='metric_mean', color='args.explainer'))
            + p9.geom_line(p9.aes(y='metric_mean', color='args.explainer'))
            + p9.geom_line(p9.aes(y='baseline_mean'), color='black', data=df_baseline)
            + p9.facet_grid("args.model ~ args.dataset", scales="free_x", labeller=annotation.model.labeller)
            + p9.scale_x_continuous(name='Masking ratio')
            + p9.scale_y_continuous(
                labels=lambda ticks: [f'{tick:.0%}' for tick in ticks],
                name='IM masked performance'
            )
            + p9.scale_color_discrete(
                breaks = annotation.explainer.breaks,
                labels = annotation.explainer.labels,
                aesthetics = ["colour", "fill"],
                name='importance measure (IM)'
            )
            + p9.scale_shape_discrete(guide=False)
            + p9.ggtitle(experiment_id))

        # Save plot, the width is the \linewidth of a collumn in the LaTeX document
        os.makedirs(f'{args.persistent_dir}/plots', exist_ok=True)
        p.save(f'{args.persistent_dir}/plots/{experiment_id}.pdf', width=3*6.30045 + 0.2, height=7, units='in')
        p.save(f'{args.persistent_dir}/plots/{experiment_id}.png', width=3*6.30045 + 0.2, height=7, units='in')
