import numpy as np
import pandas as pd


def highlight_max(data, levels=(0, 1), color="red"):
    """
    Highlight the maximum in a pandas dataframe.

    Use with df.style.apply(highlight_max,axis=<behavior>). Set axis=0 or axis=1 for per column/per row highlighting.
    Set axis=None with some levels set (e.g., (0, 1)) to highlight on some levels of a multiindex.


    Args:
        data: Dataframe or series to highlight.
        levels (tuple of int): Levels to highlight by. Ignored if data is a series.
        color: Color to set

    Returns:

    """
    attr = 'background-color: %s' % color

    if data.ndim == 1:
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:
        is_max = data.groupby(level=levels).transform('max') == data
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)


def highlight_threshold(s, threshold, column, greater=True, color="red"):
    """
    Highlight rows such that a value is greater (or less) than a threshold

    Typical use: df_out.style.apply(
                     highlight_threshold, threshold=0.1, greater=False, column=["p-value-less"], axis=1, color="green"
                 ).apply(
                     highlight_threshold, threshold=0.1, greater=False, column=["p-value-greater"], axis=1, color="red"
                 )
    Args:
        s: Series to highlight
        threshold: Value used as threshold.
        column: Name of the column to check.
        greater (bool): Whether to highlight values greater than the threshold, otherwise highlighting lesser.
        color: Color to set

    Returns:

    """
    is_max = pd.Series(data=False, index=s.index)
    if greater:
        is_max[column] = s.loc[column] >= threshold
    else:
        is_max[column] = s.loc[column] <= threshold
    return ['background-color: %s' % color if is_max.any() else '' for _ in is_max]
