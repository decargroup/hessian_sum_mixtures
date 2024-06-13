import pandas as pd
import numpy as np
from typing import Dict
import re


def highlight_min(data: pd.DataFrame, highlight_max=False):
    """
    func : function
        ``func`` should take a Series if ``axis`` in [0,1] and return a list-like
        object of same length, or a Series, not necessarily of same length, with
        valid index labels considering ``subset``.
        ``func`` should take a DataFrame if ``axis`` is ``None`` and return either
        an ndarray with the same shape or a DataFrame, not necessarily of the same
        shape, with valid index and columns labels considering ``subset``.
    """
    attr = "textbf:--rwrap;"
    ndim = data.ndim
    # ndim = len(data.index.names)
    if ndim == 1:  # Series from .apply(axis=0) or axis=1
        if highlight_max is False:
            is_min = data == data.min()
        else:
            is_min = data == data.max()
        ret_val = [attr if v else "" for v in is_min]
    else:
        if highlight_max is False:
            is_min = data.groupby(level=0).transform("min") == data
        else:
            is_min = data.groupby(level=0).transform("max") == data
        ret_val = pd.DataFrame(
            np.where(is_min, attr, ""), index=data.index, columns=data.columns
        )
    return ret_val


def format_multiindexed_df(
    df_metric: pd.DataFrame,
    floating_point_format_dict: Dict,
    column_format="|*{7}{c|}",
    min_columns=None,
    max_columns=None,
    drop_columns=["Dims", "NEES", "Method", "Run Name", "Case"],
):

    df_metric = df_metric.copy()
    if drop_columns is not None:
        df_metric.drop(columns=drop_columns, inplace=True)

    if min_columns is not None:
        df_styled: pd.DataFrame.style = df_metric.style.apply(
            highlight_min, axis=None, subset=min_columns
        )
    df_styled: pd.DataFrame.style = df_styled.apply(
        highlight_min, axis=None, subset=max_columns, highlight_max=True
    )

    df_styled = df_styled.format(floating_point_format_dict)

    latex_string = df_styled.to_latex(column_format=column_format, hrules=True)
    latex_string = latex_string.replace(
        "\\\n\end{tabular}", "\\ \hline \n\end{tabular}"
    )
    latex_string = latex_string.replace(
        "\\\n\end{tabular}", "\\ \hline \n\end{tabular}"
    )

    # latex_string = latex_string.replace("{r}", "{c|}")
    latex_string = re.sub("\\\.*rule", "\\\hline", latex_string)
    print(latex_string)
