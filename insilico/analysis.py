from math import log10, floor

import numpy as np
from scipy.stats import ttest_rel
import pandas as pd


def paired_t_test(df, col_left, col_right, common_col="seed"):
    """

    Considered calling .round(5) on output for clearer reading.

    Args:
        df (pd.DataFrame): The results of the experiment.
        col_left: Identifier of the column
        col_right:
        common_col: Identifier of the column indexing the repetitions of the experiments.

    Returns:
        pd.Dataframe: Dataframe with the mean values of the left and right column, as well as p-values of unilateral
                      tests.

    """
    # TODO: Single level not considered
    group_cols = [level.name for level in df.index.levels]
    if common_col not in group_cols:
        raise ValueError("Common column %s not found." % common_col)
    for c in [col_left, col_right]:
        if c not in df.columns:
            raise ValueError("Column %s not found" % c)
    group_cols.remove(common_col)

    df_eval = df.groupby(group_cols).agg(list)

    df_out = pd.concat(
        (
            df.groupby(group_cols).agg("mean"),
            pd.Series(
                df_eval.apply(
                    lambda row: ttest_rel(
                        row[col_left], row[col_right], alternative="less"
                    ).pvalue,
                    axis=1,
                ),
                name="p-value-less",
            ),
            pd.Series(
                df_eval.apply(
                    lambda row: ttest_rel(
                        row[col_left], row[col_right], alternative="greater"
                    ).pvalue,
                    axis=1,
                ),
                name="p-value-greater",
            ),
        ),
        axis=1,
    )
    return df_out


def format_mag_err(mag, err, sep=" ± ", increase=0, increase_ones=True):
    """
    Format a magnitude and its error as a string

    Args:
        mag (float): Value of the magnitude
        err (float): Value of the associated error
        sep (str): Characters to use to join the numbers. Include spaces if needed.
        increase (int): A number to increase (or decrease if negative) the number of significant digits.
        increase_ones (bool): Whether the number of significant digits increases by one when the leading digit is one.

    Returns:
        str: The representation of the magnitude with its error.

    """
    if np.isnan(err):
        return "%s%s%s" % (mag, sep, err)

    order = floor(log10(err))
    if increase_ones and floor(err / 10 ** order) == 1.0:  # If flag on and leading digit is 1
        order -= 1

    order -= increase
    if order < 0:
        mag = ("%%.%df" % -order) % mag
        err = ("%%.%df" % -order) % err
    else:
        mag = "%d" % round(mag, -order)
        err = "%d" % round(err, -order)

    return "%s%s%s" % (mag, sep, err)


def _format_err(row):
    out = {}
    for var in row.index.levels[0]:
        out[var] = format_mag_err(row[var]["mean"], row[var]["sem"])
    return out


def df_agg_mean(df, group_cols, raw=False):
    """
    Aggregate a dataframe to summarize it with the mean and its error

    Args:
        df (pd.DataFrame): The dataframe.
        group_cols (list of str): Columns used as index for the aggregation.
        raw (bool): If False, the result is a table of strings representing the number with its error. If True,
                    the columns will an additional level providing both the mean and its error (sem).

    Returns:
        pd.DataFrame: The summarizing dataframe

    """
    df_agg = df.groupby(group_cols).agg(['mean', 'sem'])
    if raw:
        return df_agg
    return df_agg.apply(_format_err, axis=1, result_type="expand")
