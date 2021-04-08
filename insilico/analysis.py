from scipy.stats import ttest_rel
import pandas as pd


def paired_t_test(df, col_left, col_right, common_col="seed"):
    """

    Considered calling .round(5) on output for clearer reading.

    Args:
        df (pd.Dataframe): The results of the experiment.
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
