import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

import mixtures.point_set_registration.parser_psr as parser_psr
from mixtures.point_set_registration.monte_carlo import (
    ProblemParameters,
    MonteCarloRunParameters,
)
import os
from mixtures.point_set_registration.monte_carlo import (
    load_problem_param_list,
    metric_dataframes,
    optimize_trials,
    average_metric_table,
)


def main(args):
    sns.set_theme(style="whitegrid")
    plt.style.use(args.stylesheet)
    monte_carlo_run_id = args.monte_carlo_run_id
    mc_params = MonteCarloRunParameters.from_args(args)
    print(f"Running with number of jobs: {args.n_jobs}")
    print(mc_params)
    mc_params.generate_random_transformations()
    mc_dir = os.path.join(args.top_result_dir, monte_carlo_run_id)
    problem_parameter_dir = os.path.join(mc_dir, "problem_parameters")
    opt_result_dir = os.path.join(mc_dir, "opt_results")
    Path(problem_parameter_dir).mkdir(parents=True, exist_ok=True)
    Path(opt_result_dir).mkdir(parents=True, exist_ok=True)
    csv_folder = os.path.join(mc_dir, "csv_folder")
    Path(csv_folder).mkdir(parents=True, exist_ok=True)

    fname = os.path.join(args.top_result_dir, monte_carlo_run_id, "mc_params.json")
    mc_params.to_json(fname)
    if not args.postprocess_only:
        if args.no_continue_mc:
            create_new = True
        else:
            create_new = False
            list_of_mixture_result_file_lists = []
            for mixture_approach in args.mixture_approaches:
                opt_files = list(
                    Path(os.path.join(opt_result_dir, mixture_approach)).glob("*.json")
                )
                opt_ids = [opt_file.stem for opt_file in opt_files]
                list_of_mixture_result_file_lists.append(opt_ids)
            problem_id_intersection = set.intersection(
                *map(set, list_of_mixture_result_file_lists)
            )

        problem_param_list = load_problem_param_list(
            problem_parameter_dir, create_new=create_new, mc_params=mc_params
        )

        if not args.no_continue_mc:
            # Continue Monte Carlo on problem ids we havent done yet
            problem_param_list: List[ProblemParameters] = [
                problem_param
                for problem_param in problem_param_list
                if problem_param.problem_id not in problem_id_intersection
            ]
        solver_params = solver_params_from_mc_params(mc_params)
        optimize_trials(
            args.mixture_approaches,
            opt_result_dir,
            problem_param_list,
            verbose=not args.no_verbose,
            n_jobs=args.n_jobs,
            solver_params=solver_params,
        )

    df_metric_dict = metric_dataframes(
        args.mixture_approaches, opt_result_dir, args.read_metrics_csv, csv_folder
    )
    df = average_metric_table(df_metric_dict)
    print(df)
    boxplot_dir = os.path.join(mc_dir, "boxplots")
    Path(boxplot_dir).mkdir(parents=True, exist_ok=True)

    metric_names = [
        "RMSE (deg)",
        "RMSE (m)",
        "NEES",
        "ANEES",
        "Avg Iter.",
    ]


def solver_params_from_mc_params(mc_params: MonteCarloRunParameters):
    # A bit of copypasterino. TODO: Fix.
    solver_params = {
        "solver": mc_params.solver,
        "max_iters": mc_params.max_iters,
        "step_tol": (
            mc_params.step_tol if mc_params.convergence_criterion == "step" else None
        ),
        "gradient_tol": (
            mc_params.gradient_tol
            if mc_params.convergence_criterion == "gradient"
            else None
        ),
        "ftol": (
            mc_params.ftol if mc_params.convergence_criterion == "rel_cost" else None
        ),
        "tau": mc_params.tau,
        "initial_normalization_constant_hsm": mc_params.initial_normalization_constant_hsm,
    }
    return solver_params


if __name__ == "__main__":
    parser = parser_psr.parser()
    args = parser.parse_args()
    main(args)
