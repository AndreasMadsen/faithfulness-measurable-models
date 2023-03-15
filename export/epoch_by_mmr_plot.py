
import json
import argparse
import os
import pathlib

from tqdm import tqdm
import pandas as pd
import plotnine as p9

from ecoroar.dataset import datasets
from ecoroar.plot import bootstrap_confint, annotation

def select_target_metric(partial_df):
    column_name = partial_df.loc[:, 'target_metric'].iat[0]
    return pd.Series({
        'metric': partial_df.loc[:, f'history.val_{column_name}'].iat[0]
    })


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

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    args, unknown = parser.parse_known_args()

    # TODO: figure out why only loss and accuarcy are logged by .evalaute
    dataset_mapping = pd.DataFrame([
        { 'args.dataset': dataset._name, 'target_metric': dataset._early_stopping_metric }
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
        files = sorted((args.persistent_dir / 'results').glob('masking_*.json'))
        for file in tqdm(files, desc='Loading .json files'):
            with open(file, 'r') as fp:
                try:
                    results.append(json.load(fp))
                except json.decoder.JSONDecodeError:
                    print(f'{file} has a format error')

        df = pd.json_normalize(results).explode('history', ignore_index=True)
        results = pd.json_normalize(df.pop('history')).add_prefix('history.')
        df = pd.concat([df, results], axis=1)

        # Select test metric
        df = (df
              .merge(dataset_mapping, on='args.dataset')
              .merge(model_mapping, on='args.model')
              .groupby(['args.model', 'args.seed', 'args.dataset', 'args.max_masking_ratio', 'args.max_epochs', 'args.masking_strategy',
                        'history.epoch',
                        'model_category'], group_keys=True)
              .apply(select_target_metric)
              .reset_index()
              .assign(**{'history.epoch': lambda df: df['history.epoch'] + 1}))

    if args.stage in ['preprocess']:
        os.makedirs(f'{args.persistent_dir}/pandas', exist_ok=True)
        df.to_pickle(f'{args.persistent_dir}/pandas/epoch.pd.pkl.xz')
    elif args.stage in ['plot']:
        df = pd.read_pickle(f'{args.persistent_dir}/pandas/epoch.pd.pkl.xz')

    if args.stage in ['both', 'plot']:
        # Compute confint and mean for each group

        for model_category in ['masking-ratio', 'size']:
            df_model_category = df.query('`model_category` == @model_category & \
                                          `args.masking_strategy` == "uni"')
            if df_model_category.shape[0] == 0:
                print(f'Skipping model category "{model_category}", no observations.')
                continue

            df_goal = (df_model_category
                .query('`args.max_masking_ratio` == 0')
                .groupby(['args.model', 'args.dataset', 'history.epoch', 'args.max_epochs'], group_keys=True)
                .apply(bootstrap_confint(['metric']))
                .reset_index())
            df_goal = pd.concat([
                df_goal.assign(**{
                    'args.max_masking_ratio': max_masking_ratio,
                })
                for max_masking_ratio in [0, 20, 40, 60, 80, 100]
            ])

            df_epochs = (df_model_category
                .groupby(['args.model', 'args.dataset', 'history.epoch', 'args.max_masking_ratio'], group_keys=True)
                .apply(bootstrap_confint(['metric']))
                .reset_index())

            # Generate plot
            p = (p9.ggplot(df_epochs, p9.aes(x='history.epoch'))
                + p9.geom_jitter(p9.aes(y='metric', group='args.seed', color='args.model'),
                                shape='+', alpha=0.5, width=0.25, data=df_model_category)
                + p9.geom_ribbon(p9.aes(ymin='metric_lower', ymax='metric_upper', fill='args.model'), alpha=0.35)
                + p9.geom_line(p9.aes(y='metric_mean', color='args.model'))
                + p9.geom_line(p9.aes(y='metric_mean', color='args.model'), linetype='dashed', data=df_goal)
                + p9.facet_grid("args.max_masking_ratio ~ args.dataset", scales="free_x")
                + p9.scale_x_continuous(name='Epoch')
                + p9.scale_y_continuous(
                    labels=lambda ticks: [f'{tick:.0%}' for tick in ticks],
                    name='Unmasked performance'
                )
                + p9.scale_color_discrete(
                    breaks = annotation.model.breaks,
                    labels = annotation.model.labels,
                    aesthetics = ["colour", "fill"],
                    name='Model'
                )
                + p9.scale_shape_discrete(guide=False))

            # Save plot, the width is the \linewidth of a collumn in the LaTeX document
            os.makedirs(f'{args.persistent_dir}/plots', exist_ok=True)
            p.save(f'{args.persistent_dir}/plots/epoch_by_mmr_m-{model_category}.pdf', width=3*6.30045 + 0.2, height=2*7, units='in')
            p.save(f'{args.persistent_dir}/plots/epoch_by_mmr_m-{model_category}.png', width=3*6.30045 + 0.2, height=2*7, units='in')