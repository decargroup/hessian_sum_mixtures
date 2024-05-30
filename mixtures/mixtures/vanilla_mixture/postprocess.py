from mixtures.vanilla_mixture.monte_carlo import VanillaMixtureMetric
from typing import Dict
from tqdm import tqdm
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from mixtures.vanilla_mixture.monte_carlo import (
    OptimizationResult,
)


def optimization_result_dict(mc_res_dir: str, approaches: List[str]):
    opt_result_dict = {}
    for approach in approaches:
        opt_result_dict[approach] = []
        res_dir = os.path.join(mc_res_dir, approach)
        fnames = list(Path(res_dir).iterdir())
        fnames = [str(fname) for fname in fnames]
        # fnames = fnames[:1000] # for quick debug plots
        for fname in tqdm(fnames):
            opt_result = OptimizationResult.from_json(fname)
            opt_result.compute_metrics()
            opt_result_dict[approach].append(opt_result)
    return opt_result_dict


def metric_dataframes(
    mixture_approaches: List[str],
    opt_result_dir: str,
    read_from_csv=False,
    csv_folder=None,
) -> Dict[str, pd.DataFrame]:
    metric_dict = {}
    metric_names = [
        "RMSE",
        "NEES",
        "ANEES",
        "Avg Iter.",
        "Conv. Success",
        "Dist. Optimum",
        "Time (s)",
    ]
    if not read_from_csv:
        for metric in metric_names:
            metric_dict[metric] = {}

        for mixture_approach in mixture_approaches:
            list_opt_result = list(
                Path(os.path.join(opt_result_dir, mixture_approach)).glob("*.pkl")
            )
            # list_opt_result = list_opt_result[:1000]  # for quick debug plots

            metrics = []
            print(f"Loading {len(list_opt_result)} files for {mixture_approach}")
            for fname in tqdm(list_opt_result):
                opt_result: OptimizationResult = OptimizationResult.from_pickle(fname)
                m: VanillaMixtureMetric = opt_result.compute_metrics()
                metrics.append(m)
            metric_dict["RMSE"][mixture_approach] = np.array([m.rmse for m in metrics])
            metric_dict["NEES"][mixture_approach] = np.array([m.nees for m in metrics])
            metric_dict["ANEES"][mixture_approach] = np.array(
                [m.nees / (m.dof) for m in metrics]
            )
            metric_dict["Avg Iter."][mixture_approach] = np.array(
                [m.num_iterations for m in metrics]
            )
            metric_dict["Conv. Success"][mixture_approach] = np.array(
                [m.convergence_success for m in metrics]
            )
            metric_dict["Dist. Optimum"][mixture_approach] = np.array(
                [m.distance_to_optimum for m in metrics]
            )
            metric_dict["Time (s)"][mixture_approach] = np.array(
                [m.total_time for m in metrics]
            )

        for metric in metric_names:
            df = pd.DataFrame.from_dict(metric_dict[metric])
            metric_dict[metric] = df
            df.to_csv(os.path.join(csv_folder, f"{metric}.csv"), index=False)
    else:
        for metric in metric_names:
            df = pd.read_csv(os.path.join(csv_folder, f"{metric}.csv"))
            metric_dict[metric] = df
    return metric_dict


def average_metric_table(metric_dict: Dict[str, pd.DataFrame], print_table=False):
    """_summary_

    Parameters
    ----------
    metric_dict : Dict[str, pd.DataFrame]
        metric_dict[metric_name] contains a dataframe with
        columns corresponding to the mixtures and rows corresponding to the runs.
    """
    approaches = metric_dict["RMSE"].columns
    average_metrics_dict = {}
    for approach in approaches:
        average_metrics_dict[approach] = {}
        for metric_name in ["RMSE", "NEES", "ANEES", "Avg Iter.", "Time (s)"]:
            average_metrics_dict[approach][metric_name] = np.mean(
                np.array([metric_dict[metric_name][approach]])
            )
        if not metric_dict["Conv. Success"][approach].any():
            average_metrics_dict[approach]["Converg. Succ. Rate [\%]"] = 0
        else:
            average_metrics_dict[approach]["Converg. Succ. Rate [\%]"] = (
                metric_dict["Conv. Success"][approach].value_counts(normalize=True)[
                    True
                ]
                * 100
            )

    # bop = 1
    df = pd.DataFrame.from_dict(average_metrics_dict)
    df = df.transpose()
    df_all = df[["RMSE", "NEES", "Avg Iter.", "Converg. Succ. Rate [\%]", "Time (s)"]]
    df_all = df.rename(
        columns={
            "RMSE": "RMSE (m)",
            "Converg. Succ. Rate [\%]": "Conv. Rate [\%]",
        }
    )
    df_styled = format_df(df_all)
    if print_table:
        print(df)
        print(
            df_styled.to_latex(column_format="|l|c|c|c|c|c|c|").replace(
                "\\\n", "\\ \hline\n"
            )
        )

    return df


def format_df(df: pd.DataFrame):
    cols = df.columns
    df_styled = df.style.highlight_min(
        subset=cols[cols != "Converg. Succ. Rate [\%]"],
        axis=0,
        props="textbf:--rwrap;",
    )
    df_styled = df_styled.highlight_max(
        subset=["Conv. Rate [\%]"],
        axis=0,
        props="textbf:--rwrap;",
    )
    df_styled = df_styled.format(
        {
            "RMSE (m)": "{:,.2e}".format,
            "NEES": "{:,.2f}".format,
            "ANEES": "{:,.2f}".format,
            "Avg Iterations": "{:,.2f}".format,
            "Time (s)": "{:,.2f}".format,
            "Conv. Rate [\%]": "{:,.1f}".format,
        }
    )
    return df_styled
