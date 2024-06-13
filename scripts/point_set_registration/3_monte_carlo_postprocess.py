import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import mixtures.point_set_registration.parser_psr as parser_psr
from mixtures.point_set_registration.monte_carlo import (
    metric_dataframes,
    average_metric_table,
)
import numpy as np
from mixtures.postprocess_utils import format_multiindexed_df

parser = parser_psr.parser()
parser.add_argument(
    "--monte_carlo_from_dim_string",
    help="Run id from the dimension string",
    default="psr_{}d_{}_small",
)
parser.add_argument("--dims_list", type=int, nargs="+", default=[2, 3])


def main(args):
    args.mixture_approaches = [
        "MM",
        "SM",
        "MSM",
        "HSM",
        "HSM_STD",
        "HSM_STD_NO_COMPLEX",
    ]
    sns.set_theme(style="whitegrid")
    df_list = []
    for dims in [2, 3]:
        monte_carlo_run_id = args.monte_carlo_from_dim_string.format(dims, args.solver)
        mc_dir = os.path.join(args.top_result_dir, monte_carlo_run_id)
        opt_result_dir = os.path.join(mc_dir, "opt_results")
        csv_folder = os.path.join(mc_dir, "csv_folder")
        Path(csv_folder).mkdir(parents=True, exist_ok=True)

        df_metric_dict = metric_dataframes(
            args.mixture_approaches, opt_result_dir, args.read_metrics_csv, csv_folder
        )
        df = average_metric_table(df_metric_dict)
        df["Dims"] = f"{dims}D"
        df["Method"] = df.index
        df["Run Name"] = df.index
        df["Case"] = df.index
        df_list.append(df)
    df = pd.concat(df_list)
    print(df.columns)
    index = pd.MultiIndex.from_frame(df[["Dims", "Method"]])
    df = df.set_index(index)
    df = df.rename(
        columns={
            "Avg Iterations": "Avg Iter.",
        }
    )
    print(df)
    format_multiindexed_df(
        df,
        {
            "RMSE (deg)": "{:,.2f}".format,
            "RMSE (m)": "{:,.2f}".format,
            "ANEES": "{:,.2f}".format,
            "Time (s)": "{:,.2f}".format,
            "Avg Iter.": "{:,.2f}".format,
        },
        min_columns=["RMSE (deg)", "RMSE (m)", "ANEES", "Avg Iter.", "Time (s)"],
        max_columns=None,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
