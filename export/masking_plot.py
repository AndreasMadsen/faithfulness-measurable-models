
import glob
import json
import argparse
import os
import os.path as path

from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy.stats
import plotnine as p9

from ecoroar.dataset import datasets


def select_test_metric(partial_df):
    column_name = partial_df.loc[:, 'test_metric'].iat[0]
    return pd.Series({
        'metric': partial_df.loc[:, f'results.{column_name}'].iat[0]
    })


def ratio_confint(column_names):
    """Implementes a ratio-confidence interval
    This one uses bootstrapping.
    Method proposed here: https://stats.stackexchange.com/questions/263516
    """
    def agg(partial_df):
        summary = dict()
        partial_df = partial_df.reset_index()

        for column_name in column_names:
            x = partial_df.loc[:, column_name].to_numpy()
            mean = np.mean(x)

            if np.all(x[0] == x):
                lower = mean
                upper = mean
            else:
                res = scipy.stats.bootstrap(
                    (x, ), np.mean,
                    confidence_level=0.95, random_state=np.random.default_rng(0)
                )
                lower = res.confidence_interval.low
                upper = res.confidence_interval.high

            summary.update({
                f'{column_name}_lower': lower,
                f'{column_name}_mean': mean,
                f'{column_name}_upper': upper,
                f'{column_name}_n': len(x)
            })

        return pd.Series(summary)
    return agg


thisdir = path.dirname(path.realpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--persistent-dir',
                    action='store',
                    default=path.realpath(path.join(thisdir, '..')),
                    type=str,
                    help='Directory where all persistent data will be stored')
parser.add_argument('--stage',
                    action='store',
                    default='both',
                    type=str,
                    choices=['preprocess', 'plot', 'both'],
                    help='Which export stage should be performed. Mostly just useful for debugging.')

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    args, unknown = parser.parse_known_args()

    # TODO: figure out why only loss and accuarcy are logged by .evalaute
    dataset_mapping = pd.DataFrame([
        { 'args.dataset': dataset._name, 'test_metric': 'accuracy' } # dataset._early_stopping_metric }
        for dataset in datasets.values()
    ])
    model_mapping = pd.DataFrame([
        { 'args.model': 'roberta-m15', 'model_category': 'masking-ratio' },
        { 'args.model': 'roberta-m20', 'model_category': 'masking-ratio' },
        { 'args.model': 'roberta-m30', 'model_category': 'masking-ratio' },
        { 'args.model': 'roberta-m40', 'model_category': 'masking-ratio' },
        { 'args.model': 'roberta-m50', 'model_category': 'masking-ratio' },
        { 'args.model': 'roberta-sb', 'model_category': 'size' },
        { 'args.model': 'roberta-sl', 'model_category': 'size' }
    ])

    if args.stage in ['both', 'preprocess']:
        # Read JSON files into dataframe
        results = []
        files = glob.glob(f'{args.persistent_dir}/results/masking_*.json')
        for file in tqdm(files, desc='Loading .json files'):
            with open(file, 'r') as fp:
                try:
                    results.append(json.load(fp))
                except json.decoder.JSONDecodeError:
                    print(f'{file} has a format error')

        df = pd.json_normalize(results).explode('results', ignore_index=True)
        results = pd.json_normalize(df.pop('results')).add_prefix('results.')
        df = pd.concat([df, results], axis=1)

        # Select test metric
        df = (df
              .merge(dataset_mapping, on='args.dataset')
              .merge(model_mapping, on='args.model')
              .groupby(['args.model', 'args.seed', 'args.dataset', 'args.max_masking_ratio',
                        'results.masking_ratio',
                        'model_category'])
              .apply(select_test_metric)
              .reset_index())

    if args.stage in ['preprocess']:
        os.makedirs(f'{args.persistent_dir}/pandas', exist_ok=True)
        df.to_pickle(f'{args.persistent_dir}/pandas/masking.pd.pkl.xz')
    elif args.stage in ['plot']:
        df = pd.read_pickle(f'{args.persistent_dir}/pandas/masking.pd.pkl.xz')

    if args.stage in ['both', 'plot']:
        # Compute confint and mean for each group

        for model_category in ['masking-ratio', 'size']:
            df_subset = df.loc[(df['results.masking_ratio'] == 0.0) & (df['model_category'] == model_category), :]
            df_plot = (df_subset
                    .groupby(['args.model', 'args.dataset', 'args.max_masking_ratio'])
                    .apply(ratio_confint(['metric']))
                    .reset_index()
            )

            # Generate plot
            p = (p9.ggplot(df_plot, p9.aes(x='args.max_masking_ratio'))
                + p9.geom_ribbon(p9.aes(ymin='metric_lower', ymax='metric_upper', fill='args.model'), alpha=0.35)
                + p9.geom_line(p9.aes(y='metric_mean', color='args.model'))
                + p9.geom_point(p9.aes(y='metric_mean', color='args.model', shape='args.model'))
                + p9.geom_jitter(p9.aes(y='metric', group='args.seed', shape='args.dataset', color='args.model'),
                                alpha=0.3, width=1, data=df_subset)
                + p9.facet_grid("args.dataset ~", scales="free_y")
                + p9.labs(y='Performance', shape='', x='Max masking ratio')
                + p9.scale_y_continuous(labels=lambda ticks: [f'{tick:.0%}' for tick in ticks])
                + p9.scale_x_continuous(labels=lambda ticks: [f'{tick:.0f}%' for tick in ticks])
                + p9.scale_shape_discrete(guide=False))

            # Save plot, the width is the \linewidth of a collumn in the LaTeX document
            os.makedirs(f'{args.persistent_dir}/plots', exist_ok=True)
            p.save(f'{args.persistent_dir}/plots/masking_max-ratio_m-{model_category}.pdf', width=6.30045 + 0.2, height=7, units='in')
            p.save(f'{args.persistent_dir}/plots/masking_max-ratio_m-{model_category}.png', width=6.30045 + 0.2, height=7, units='in')
