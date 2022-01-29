
import glob
import json
import argparse
import os
import os.path as path

from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy
import plotnine as p9


def select_test_metric(partial_df):
    column_name = partial_df.loc[:, 'test_metric'].iat[0]
    return pd.Series({
        'metric': partial_df.loc[:, column_name].iat[0]
    })


def ratio_confint(column_names):
    """Implementes a ratio-confidence interval

    The idea is to project to logits space, then assume a normal distribution,
    and then project back to the inital space.

    Method proposed here: https://stats.stackexchange.com/questions/263516
    """
    def agg(partial_df):
        summary = dict()

        for column_name in column_names:
            x = partial_df.loc[:, column_name]
            logits = scipy.special.logit(x)
            mean = np.mean(logits)
            sem = scipy.stats.sem(logits)
            lower, upper = scipy.stats.t.interval(0.95, len(x) - 1,
                                                  loc=mean,
                                                  scale=sem)

            lower = scipy.special.expit(lower)
            mean = scipy.special.expit(mean)
            upper = scipy.special.expit(upper)
            if np.isnan(sem) or sem == 0:
                lower, upper = mean, mean

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

    dataset_mapping = pd.DataFrame([
        {'dataset': 'imdb', 'dataset_pretty': 'IMDB', 'test_metric': 'macro_f1'}
    ])

    if args.stage in ['both', 'preprocess']:
        # Read JSON files into dataframe
        results = []
        files = glob.glob(f'{args.persistent_dir}/results/masking-effect/*.json')
        for file in tqdm(files, desc='Loading .json files'):
            with open(file, 'r') as fp:
                try:
                    results.append(json.load(fp))
                except json.decoder.JSONDecodeError:
                    print(f'{file} has a format error')
        df = pd.DataFrame(results)

        # Compute confint and mean for each group
        df = (df
              .merge(dataset_mapping, on='dataset')
              .drop(['dataset'], axis=1)
              .groupby(['seed', 'dataset_pretty', 'max_masking_ratio'])
              .apply(select_test_metric))

    if args.stage in ['preprocess']:
        os.makedirs(f'{args.persistent_dir}/pandas', exist_ok=True)
        df.to_pickle(f'{args.persistent_dir}/pandas/masking-effect.pd.pkl.xz')
    if args.stage in ['plot']:
        df = pd.read_pickle(f'{args.persistent_dir}/pandas/masking-effect.pd.pkl.xz')

    if args.stage in ['both', 'plot']:
        # Generate result plots
        df_plot = df.groupby(['dataset_pretty', 'max_masking_ratio']).apply(ratio_confint(['metric']))

        # Generate plot
        p = (p9.ggplot(df_plot.reset_index(), p9.aes(x='max_masking_ratio'))
             + p9.geom_ribbon(p9.aes(ymin='metric_lower', ymax='metric_upper', fill='dataset_pretty'), alpha=0.35)
             + p9.geom_line(p9.aes(y='metric_mean', color='dataset_pretty'))
             + p9.geom_point(p9.aes(y='metric_mean', color='dataset_pretty', shape='dataset_pretty'))
             + p9.geom_jitter(p9.aes(y='metric', group='seed', shape='dataset_pretty'),
                              color='black', alpha=0.3, width=1, data=df.reset_index())
             + p9.labs(y='Macro F1', shape='', x='Max masking ratio')
             + p9.scale_y_continuous(labels=lambda ticks: [f'{tick:.0%}' for tick in ticks])
             + p9.scale_x_continuous(labels=lambda ticks: [f'{tick:.0f}%' for tick in ticks]))

        # Save plot, the width is the \linewidth of a collumn in the LaTeX document
        os.makedirs(f'{args.persistent_dir}/plots', exist_ok=True)
        p.save(f'{args.persistent_dir}/plots/imdb-masking.pdf', width=6.30045 + 0.2, height=7, units='in')
        p.save(f'{args.persistent_dir}/plots/imdb-masking.png', width=6.30045 + 0.2, height=7, units='in')
