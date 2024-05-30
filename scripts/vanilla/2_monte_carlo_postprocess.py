import os
from mixtures.vanilla_mixture.parser_vanilla import get_parser_vanilla
from mixtures.vanilla_mixture.postprocess import (
    average_metric_table,
    metric_dataframes,
)
from pathlib import Path
import seaborn as sns
import pandas as pd
from mixtures.postprocess_utils import format_multiindexed_df

parser = get_parser_vanilla()
parser.add_argument(
    "--monte_carlo_from_dim_strings",
    help="Run id from the dimension string",
    nargs="+",
    # default=[
    # "thousand_mix_thousand_starts_{}d_near",
    # "thousand_mix_thousand_starts_{}d_far",
    # ],
    # default=[
    #     "thousand_mix_thousand_starts_{}d_near",
    #     "thousand_mix_thousand_starts_{}d_far",
    # ],
    # default=["test_{}"],
    default=[
        "vanilla_many_components_{}d_near_step",
        "vanilla_many_components_{}d_far_step",
    ],
)
parser.add_argument(
    "--mc_run_names",
    nargs="+",
    # default=["test"],
    default=["Near"],
    # default=["Far"],
)

parser.add_argument(
    "--dims_list",
    type=int,
    nargs="+",
    # default=[1, 2]
    default=[1, 2],
)


def main(args):
    sns.set_theme(style="whitegrid")
    df_list = []
    # args.read_metrics_csv = True
    # mc_run_name Near
    # monte_carlo_run_id_string vanilla_many_components_{}d_near_ste
    for mc_run_name, monte_carlo_run_id_string in zip(
        args.mc_run_names, args.monte_carlo_from_dim_strings
    ):
        for dims in args.dims_list:
            monte_carlo_run_id_string: str = monte_carlo_run_id_string
            monte_carlo_run_id = monte_carlo_run_id_string.format(dims)
            mc_dir = os.path.join(args.top_result_dir, monte_carlo_run_id)
            opt_result_dir = os.path.join(mc_dir)
            csv_folder = os.path.join(mc_dir, "csv_folder")
            Path(csv_folder).mkdir(parents=True, exist_ok=True)

            df_metric_dict = metric_dataframes(
                args.mixture_approaches,
                opt_result_dir,
                args.read_metrics_csv,
                csv_folder,
            )
            df = average_metric_table(df_metric_dict)
            df["Dims"] = f"{dims}D"
            df["Method"] = df.index

            df.loc[df["Method"] == "HSM_STD_NO_COMPLEX", "Method"] = "NLS HSM"

            df["Run Name"] = mc_run_name
            df["Case"] = df["Dims"] + " " + df["Run Name"]
            df_list.append(df)
    df = pd.concat(df_list)
    index = pd.MultiIndex.from_frame(df[["Case", "Method"]])
    df = df.set_index(index)
    df = df.rename(
        columns={
            "Converg. Succ. Rate [\%]": "Succ. Rate [\%]",
            "Avg Iter.": "Iterations",
        }
    )

    print(df)
    format_multiindexed_df(
        df,
        {
            "RMSE": "{:,.2e}".format,
            "ANEES": "{:,.2e}".format,
            "Iterations": "{:,.2f}".format,
            "Succ. Rate [\%]": "{:,.1f}".format,
            "Time (s)": "{:,.2e}".format,
        },
        column_format="|*{5}{c|}",
        min_columns=["RMSE", "Iterations"],
        max_columns=["Succ. Rate [\%]"],
        drop_columns=["Dims", "NEES", "ANEES", "Method", "Run Name", "Case"],
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
