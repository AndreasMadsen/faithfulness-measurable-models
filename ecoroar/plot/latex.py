
_backslash = '\\'

def ci_formatter(column_names=[]):
    """Format upper, mean, and lower columns to latex.

    Args:
        column_names (List[str]): The columns to aggregate
    """
    def applier(df):

        add_columns = dict()
        for name in column_names:
            add_columns[f'{name}_format'] = [
                f'${{{mean*100:.0f}{_backslash}%}}^{{+{(upper - mean)*100:.1f}}}_{{-{(mean - lower)*100:.1f}}}$'
                for upper, mean, lower
                in zip(df[f'{name}_upper'], df[f'{name}_mean'], df[f'{name}_lower'])
            ]

        return df.assign(**add_columns)
    return applier
