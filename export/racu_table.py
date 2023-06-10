
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
        metric = df.reindex(cols, axis=1).to_numpy()[np.arange(len(df)), idx]
    )


def compute_acu(df):
    df_sorted = df.sort_values(by=['results.masking_ratio'])
    masking_ratio = df_sorted['results.masking_ratio'].to_numpy()
    measure = df_sorted['metric'].to_numpy()
    baseline = df_sorted['baseline'].to_numpy()

    # explainer
    y_diff = baseline - measure
    x_diff = np.diff(masking_ratio)
    areas = (y_diff[1:] + y_diff[0:-1]) * 0.5 * x_diff
    total = np.sum(areas)

    # baseline
    y_diff = baseline - baseline[-1]
    x_diff = np.diff(masking_ratio)
    areas = (y_diff[1:] + y_diff[0:-1]) * 0.5 * x_diff
    max_area = np.sum(areas)

    return pd.Series({
        'acu': total,
        'racu': total / max_area
    })

def tex_format_ci(mean, lower, upper):
    if np.isnan(mean):
        return '--'

    return f'${mean*100:.1f}_{{{-(mean - lower)*100:.1f}}}^{{+{(upper - mean)*100:.1f}}}$'

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
                    choices=['paper', 'appendix'],
                    help='The dimentions and format of the plot.')
parser.add_argument('--datasets',
                    action='store',
                    nargs='+',
                    default=list(datasets.keys()),
                    choices=datasets.keys(),
                    type=str,
                    help='The datasets to show')
parser.add_argument('--explainers',
                    action='store',
                    nargs='+',
                    default=list(explainers.keys()),
                    choices=explainers.keys(),
                    type=str,
                    help='The explainers to show')
parser.add_argument('--page',
                    action='store',
                    default=None,
                    type=str,
                    help='The page name')
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
            'target_metric': dataset._early_stopping_metric if args.performance_metric == 'primary' else args.performance_metric
        }
        for dataset in datasets.values()
    ])

    recursive_roar = [
        { 'model': 'roberta-sb', 'dataset': 'MIMIC-a', 'explainer': 'grad-l2', 'mean': 18.2, 'upper': 11.8, 'lower': -13.8 },
        { 'model': 'roberta-sb', 'dataset': 'MIMIC-a', 'explainer': 'inp-grad-abs', 'mean': 8.8, 'upper': 22.7, 'lower': -22.8 },
        { 'model': 'roberta-sb', 'dataset': 'MIMIC-a', 'explainer': 'int-grad-abs', 'mean': 12.5, 'upper': 11.3, 'lower': -7.0 },

        { 'model': 'roberta-sb', 'dataset': 'MIMIC-d', 'explainer': 'grad-l2', 'mean': 57.9, 'upper': 14.4, 'lower': -19.8 },
        { 'model': 'roberta-sb', 'dataset': 'MIMIC-d', 'explainer': 'inp-grad-abs', 'mean': 53.4, 'upper': 23.2, 'lower': -29.3 },
        { 'model': 'roberta-sb', 'dataset': 'MIMIC-d', 'explainer': 'int-grad-abs', 'mean': 26.1, 'upper': 12.0, 'lower': -25.1 },

        { 'model': 'roberta-sb', 'dataset': 'IMDB', 'explainer': 'grad-l2', 'mean': 25.4, 'upper': 3.1, 'lower': -2.0 },
        { 'model': 'roberta-sb', 'dataset': 'IMDB', 'explainer': 'inp-grad-abs', 'mean': 16.9, 'upper': 1.1, 'lower': -3.0 },
        { 'model': 'roberta-sb', 'dataset': 'IMDB', 'explainer': 'int-grad-abs', 'mean': 35.1, 'upper': 2.4, 'lower': -1.7 },

        { 'model': 'roberta-sb', 'dataset': 'SNLI', 'explainer': 'grad-l2', 'mean': 50.7, 'upper': 1.1, 'lower': -0.8 },
        { 'model': 'roberta-sb', 'dataset': 'SNLI', 'explainer': 'inp-grad-abs', 'mean': 41.0, 'upper': 0.4, 'lower': -0.5 },
        { 'model': 'roberta-sb', 'dataset': 'SNLI', 'explainer': 'int-grad-abs', 'mean': 56.7, 'upper': 1.0, 'lower': -1.1 },

        { 'model': 'roberta-sb', 'dataset': 'SST2', 'explainer': 'grad-l2', 'mean': 26.1, 'upper': 1.6, 'lower': -2.2 },
        { 'model': 'roberta-sb', 'dataset': 'SST2', 'explainer': 'inp-grad-abs', 'mean': 18.6, 'upper': 4.1, 'lower': -4.6 },
        { 'model': 'roberta-sb', 'dataset': 'SST2', 'explainer': 'int-grad-abs', 'mean': 32.9, 'upper': 1.8, 'lower': -1.5 },

        { 'model': 'roberta-sb', 'dataset': 'bAbI-1', 'explainer': 'grad-l2', 'mean': 64.2, 'upper': 2.6, 'lower': -2.6 },
        { 'model': 'roberta-sb', 'dataset': 'bAbI-1', 'explainer': 'inp-grad-abs', 'mean': 52.1, 'upper': 1.8, 'lower': -3.7 },
        { 'model': 'roberta-sb', 'dataset': 'bAbI-1', 'explainer': 'int-grad-abs', 'mean': 48.2, 'upper': 4.1, 'lower': -5.7 },

        { 'model': 'roberta-sb', 'dataset': 'bAbI-2', 'explainer': 'grad-l2', 'mean': 57.8, 'upper': 2.0, 'lower': -2.0 },
        { 'model': 'roberta-sb', 'dataset': 'bAbI-2', 'explainer': 'inp-grad-abs', 'mean': 48.1, 'upper': 3.2, 'lower': -3.5 },
        { 'model': 'roberta-sb', 'dataset': 'bAbI-2', 'explainer': 'int-grad-abs', 'mean': 42.0, 'upper': 3.8, 'lower': -4.8 },

        { 'model': 'roberta-sb', 'dataset': 'bAbI-3', 'explainer': 'grad-l2', 'mean': 34.0, 'upper': 14.6, 'lower': -15.1 },
        { 'model': 'roberta-sb', 'dataset': 'bAbI-3', 'explainer': 'inp-grad-abs', 'mean': 22.4, 'upper': 15.9, 'lower': -12.4 },
        { 'model': 'roberta-sb', 'dataset': 'bAbI-3', 'explainer': 'int-grad-abs', 'mean': -27.9, 'upper': 18.0, 'lower': -49.1 },
    ]

    experiment_id = generate_experiment_id('racu',
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
        rroar_df = pd.DataFrame(recursive_roar)
        rroar_df = (rroar_df
            .assign(**{
                'mean': rroar_df['mean'] / 100,
                'lower': (rroar_df['mean'] + rroar_df['lower']) / 100,
                'upper': (rroar_df['mean'] + rroar_df['upper']) / 100
            })
            .rename(columns={
                'model': 'args.model',
                'dataset': 'args.dataset',
                'explainer': 'args.explainer',
                'mean': 'rroar_racu_mean',
                'upper': 'rroar_racu_upper',
                'lower': 'rroar_racu_lower'
            })
        )

        df_tab = (df
            .query('`args.explainer` != "rand"')
            .merge(df.query('`args.explainer` == "rand"').drop(columns=['args.explainer']).rename(columns={'metric': 'baseline'}),
                   on=['args.seed', 'args.model', 'args.dataset', 'results.masking_ratio'])
            .groupby(['args.seed', 'args.model', 'args.dataset', 'args.explainer'], group_keys=True)
            .apply(compute_acu)
            .reset_index()
            .groupby(['args.model', 'args.dataset', 'args.explainer'], group_keys=True)
            .apply(bootstrap_confint(['acu', 'racu']))
            .reset_index()
            .merge(rroar_df, on=['args.model', 'args.dataset', 'args.explainer'], how='left')
            .drop(columns=['args.model'])
            .transform(annotate_explainer)
            .set_index(['args.dataset', 'args.explainer']))

        explainers = [x for x in args.explainers if x in set(df['args.explainer'].unique()) and x != 'rand']

        os.makedirs(args.persistent_dir / 'tables' / args.format, exist_ok=True)
        with open(args.persistent_dir / 'tables' / args.format / f'{experiment_id}.tex', 'w') as fp:
            print(r'\begin{tabular}[t]{llccc}', file=fp)
            print(r'\toprule', file=fp)
            print(r'& & \multicolumn{3}{c}{Faithfulness [\%]}  \\', file=fp)
            print(r'\cmidrule(r){3-5}', file=fp)
            print(r'Dataset & IM & \multicolumn{2}{c}{Our} & R-ROAR \\', file=fp)
            print(r'\cmidrule(r){3-4}', file=fp)
            print(r'& & ACU & RACU & RACU \\', file=fp)
            print(r'\midrule', file=fp)
            first_row = True
            for dataset_name in args.datasets:
                first_dataset_row = True
                for im_name in explainers:
                    try:
                        dur = df_tab.loc[pd.IndexSlice[dataset_name, im_name], :]
                        our_racu =  tex_format_ci(dur['racu_mean'], dur['racu_lower'], dur['racu_upper'])
                        our_acu = tex_format_ci(dur['acu_mean'], dur['acu_lower'], dur['acu_upper'])
                        rroar_racu = tex_format_ci(dur['rroar_racu_mean'], dur['rroar_racu_lower'], dur['rroar_racu_upper'])
                    except KeyError:
                        our_racu =  tex_format_ci(np.nan, np.nan, np.nan)
                        our_acu = tex_format_ci(np.nan, np.nan, np.nan)
                        rroar_racu = tex_format_ci(np.nan, np.nan, np.nan)

                    if first_dataset_row and not first_row:
                        print(r'\cmidrule{1-5}', file=fp)
                    if first_row:
                        first_row = False

                    if first_dataset_row:
                        first_dataset_row = False
                        dataset_str = f'\multirow[c]{{{len(explainers)}}}{{*}}{{{annotation.dataset.labeller(dataset_name)}}}'
                    else:
                        dataset_str = ''

                    print(f'{dataset_str} & {annotation.explainer.labeller(im_name)} & {our_acu} & {our_racu} & {rroar_racu} \\\\', file=fp)

            print(r'\bottomrule', file=fp)
            print(r'\end{tabular}', file=fp)
