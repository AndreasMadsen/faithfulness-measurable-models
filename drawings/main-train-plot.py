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

    # bAbI-3, roberta-sb
    # masked = loss * 2 - unmasked
    df_loss = pd.DataFrame([
        { "epoch":  0, "dataset": "masked",   "loss": 1.80 },
        { "epoch":  1, "dataset": "masked",   "loss": 1.792553 },
        { "epoch":  2, "dataset": "masked",   "loss": 1.791818 },
        { "epoch":  3, "dataset": "masked",   "loss": 1.791958 },
        { "epoch":  4, "dataset": "masked",   "loss": 1.779657 },
        { "epoch":  5, "dataset": "masked",   "loss": 1.748193 },
        { "epoch":  6, "dataset": "masked",   "loss": 1.719258 },
        { "epoch":  7, "dataset": "masked",   "loss": 1.632122 },
        { "epoch":  8, "dataset": "masked",   "loss": 1.557684 },
        { "epoch":  9, "dataset": "masked",   "loss": 1.492670 },
        { "epoch": 10, "dataset": "masked",   "loss": 1.438341 },
        { "epoch": 11, "dataset": "masked",   "loss": 1.429608 },
        { "epoch": 12, "dataset": "masked",   "loss": 1.408677 },
        { "epoch": 13, "dataset": "masked",   "loss": 1.400708 },
        { "epoch": 14, "dataset": "masked",   "loss": 1.401820 },
        { "epoch": 15, "dataset": "masked",   "loss": 1.396419 },
        { "epoch": 16, "dataset": "masked",   "loss": 1.394571 },
        { "epoch": 17, "dataset": "masked",   "loss": 1.392456 },
        { "epoch": 18, "dataset": "masked",   "loss": 1.376506 },
        { "epoch": 19, "dataset": "masked",   "loss": 1.403935 },
        { "epoch": 20, "dataset": "masked",   "loss": 1.394827 },
        { "epoch":  0, "dataset": "unmasked", "loss": 1.80 },
        { "epoch":  1, "dataset": "unmasked", "loss": 1.792559 },
        { "epoch":  2, "dataset": "unmasked", "loss": 1.791600 },
        { "epoch":  3, "dataset": "unmasked", "loss": 1.791484 },
        { "epoch":  4, "dataset": "unmasked", "loss": 1.763541 },
        { "epoch":  5, "dataset": "unmasked", "loss": 1.662847 },
        { "epoch":  6, "dataset": "unmasked", "loss": 1.538212 },
        { "epoch":  7, "dataset": "unmasked", "loss": 1.247706 },
        { "epoch":  8, "dataset": "unmasked", "loss": 0.947434 },
        { "epoch":  9, "dataset": "unmasked", "loss": 0.612304 },
        { "epoch": 10, "dataset": "unmasked", "loss": 0.369407 },
        { "epoch": 11, "dataset": "unmasked", "loss": 0.298140 },
        { "epoch": 12, "dataset": "unmasked", "loss": 0.302567 },
        { "epoch": 13, "dataset": "unmasked", "loss": 0.278446 },
        { "epoch": 14, "dataset": "unmasked", "loss": 0.273084 },
        { "epoch": 15, "dataset": "unmasked", "loss": 0.258463 },
        { "epoch": 16, "dataset": "unmasked", "loss": 0.259473 },
        { "epoch": 17, "dataset": "unmasked", "loss": 0.270334 },
        { "epoch": 18, "dataset": "unmasked", "loss": 0.248268 },
        { "epoch": 19, "dataset": "unmasked", "loss": 0.259589 },
        { "epoch": 20, "dataset": "unmasked", "loss": 0.254693 }
    ])

    p = (p9.ggplot(df_loss, p9.aes(x='epoch'))
        + p9.geom_line(p9.aes(y='loss', color='dataset'))
        + p9.scale_y_continuous(name='Loss', limits=[0, None])
        + p9.scale_x_continuous(name='Epoch', limits=[0, 20])
        + p9.scale_color_discrete(
            breaks = ['masked', 'unmasked']
        )
        + p9.theme(
                plot_margin=0,
                text=p9.element_text(size=7, fontname='Times New Roman'),
                axis_title=p9.element_text(size=9),
                legend_title=p9.element_blank())
    )

    p.save(args.persistent_dir / 'drawings' / 'main-train.pdf', width=1.4, height=0.9, units='in')
