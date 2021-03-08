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
