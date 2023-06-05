
import argparse
import pathlib

import plotnine as p9
import scipy.stats
import numpy as np
import pandas as pd

def select_target_metric(df):
    idx, cols = pd.factorize('results.' + df.loc[:, 'target_metric'])
    return df.assign(
        metric = df.reindex(cols, axis=1).to_numpy()[np.arange(len(df)), idx]
    )

parser = argparse.ArgumentParser(
    description = 'Plots the 0% masking test performance given different training masking ratios')
parser.add_argument('--persistent-dir',
                    action='store',
                    default=pathlib.Path(__file__).absolute().parent.parent,
                    type=pathlib.Path,
                    help='Directory where all persistent data will be stored')

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    args, unknown = parser.parse_known_args()

    samples = np.concatenate([
        scipy.stats.truncnorm.rvs(-7.5, 1.5, loc=-3, scale=1.0, size=50, random_state=0),
        scipy.stats.truncnorm.rvs(-3, 3, loc=1, scale=0.5, size=50, random_state=0)
    ])

    df = pd.DataFrame({ 'x': samples })

    p = (p9.ggplot(df, p9.aes(x='x'))
        + p9.stat_ecdf()
        + p9.stat_ecdf(p9.aes(ymin=0, ymax=p9.after_stat('y')), geom="ribbon", alpha=0.1)
        + p9.scale_x_continuous(name='Internal representation')
        + p9.scale_y_continuous(
            labels=lambda ticks: [f'{tick:.0%}' for tick in ticks],
            name='CDF'
        )
        + p9.theme(
            plot_margin=0,
            text=p9.element_text(size=7, fontname='Times New Roman'),
            axis_title=p9.element_text(size=9)
        ))

    p.save(args.persistent_dir / 'drawings' / 'main-cdf.pdf', width=1.4, height=0.9, units='in')
