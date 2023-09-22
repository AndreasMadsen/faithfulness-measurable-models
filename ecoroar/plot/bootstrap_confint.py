
import pandas as pd
import numpy as np
import scipy.stats
from typing import Callable, List


def bootstrap_confint(column_names: List[str], seed: int = 0, aggregator: Callable[[np.ndarray], np.ndarray] = np.mean):
    """Implementes bootstrap based confidence interval.

    Such method is particularly useful for performance metrics which are definetly not
    normally distributed.

    Args:
        column_names (List[str]): The columns to aggregate
        seed (int, optional): The seed to use for bootstraping. Defaults to 0.
        aggregator (Callable[[np.ndarray], np.ndarray], optional):
            The aggregator function maps a vector to a scalar.
            Defaults to `np.mean`.
    """
    def agg(partial_df):
        summary = dict()
        partial_df = partial_df.reset_index()

        for column_name in column_names:
            x = partial_df.loc[:, column_name].to_numpy()
            mean = aggregator(x)

            if np.all(x[0] == x):
                lower = mean
                upper = mean
            else:
                res = scipy.stats.bootstrap(
                    (x, ), aggregator,
                    confidence_level=0.95, random_state=np.random.default_rng(seed)
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
