
import argparse
import pathlib

import pandas as pd
import plotnine as p9

parser = argparse.ArgumentParser()
parser.add_argument('--persistent-dir',
                    action='store',
                    default=pathlib.Path(__file__).absolute().parent.parent,
                    type=pathlib.Path,
                    help='Directory where all persistent data will be stored')

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    args, unknown = parser.parse_known_args()

    df_line = pd.DataFrame([
        {'importance_measure': 'Random', 'metric': 0.90, 'k': 0 },
        {'importance_measure': 'Random', 'metric': 0.72, 'k': 20 },
        {'importance_measure': 'Random', 'metric': 0.55, 'k': 40 },
        {'importance_measure': 'Random', 'metric': 0.45, 'k': 60 },
        {'importance_measure': 'Random', 'metric': 0.29, 'k': 80 },
        {'importance_measure': 'Random', 'metric': 0.20, 'k': 100 },

        {'importance_measure': 'Explanation', 'metric': 0.90, 'k': 0 },
        {'importance_measure': 'Explanation', 'metric': 0.45, 'k': 20 },
        {'importance_measure': 'Explanation', 'metric': 0.29, 'k': 40 },
        {'importance_measure': 'Explanation', 'metric': 0.27, 'k': 60 },
        {'importance_measure': 'Explanation', 'metric': 0.20, 'k': 80 },
        {'importance_measure': 'Explanation', 'metric': 0.20, 'k': 100 },
    ])

    df_area = pd.DataFrame([
        {'area': 'Faithfullness', 'metric_upper': 0.90, 'metric_lower': 0.90, 'k': 0 },
        {'area': 'Faithfullness', 'metric_upper': 0.72, 'metric_lower': 0.45, 'k': 20 },
        {'area': 'Faithfullness', 'metric_upper': 0.55, 'metric_lower': 0.29, 'k': 40 },
        {'area': 'Faithfullness', 'metric_upper': 0.45, 'metric_lower': 0.27, 'k': 60 },
        {'area': 'Faithfullness', 'metric_upper': 0.29, 'metric_lower': 0.20, 'k': 80 },
        {'area': 'Faithfullness', 'metric_upper': 0.20, 'metric_lower': 0.20, 'k': 100 },
    ])

    p = (p9.ggplot(mapping=p9.aes(x='k'))
        + p9.geom_ribbon(p9.aes(ymin='metric_lower', ymax='metric_upper', fill='area'), data=df_area, alpha=0.3)
        + p9.geom_line(p9.aes(y='metric', color='importance_measure', shape='importance_measure'), data=df_line)
        + p9.geom_point(p9.aes(y='metric', color='importance_measure', shape='importance_measure'), data=df_line)
        + p9.labs(y='', color='Explanation', shape='Explanation', fill='Area')
        + p9.scale_y_continuous(name='Performance', limits=[0, 1], labels = lambda ticks: [f'{tick:.0%}' for tick in ticks])
        + p9.scale_x_continuous(name='% tokens masked', breaks=range(0, 101, 20))
        + p9.scale_color_manual(
            values = ['#F8766D', '#E76BF3'],
            breaks = ['Explanation', 'Random']
        )
        + p9.scale_shape_manual(
            values = ['D', 'v'],
            breaks = ['Explanation', 'Random']
        )
        + p9.guides(fill=False)
        + p9.theme(
                plot_margin=0,
                text=p9.element_text(size=7, fontname='Times New Roman'),
                axis_title=p9.element_text(size=9),
                legend_title=p9.element_blank())
    )

    p.save(args.persistent_dir / 'drawings' / 'main-faithfulness.pdf', width=1.4, height=0.9, units='in')
