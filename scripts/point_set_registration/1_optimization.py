import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import mixtures.point_set_registration.parser_psr as parser_psr
from mixtures.point_set_registration.monte_carlo import (
    MonteCarloRunParameters,
    OptimizationResult,
    ProblemParameters,
    load_problem_param_list,
)
from mixtures.point_set_registration.plotting_psr import (
    draw_reference_frame,
    plot_problem_setup,
)
from mixtures.point_set_registration.point_set_registration import (
    solve_psr_problem,
)
from navlie.lib.states import SE2State, SE3State
from pymlg import SE2, SE3, SO2


def main(args):
    test_saving_dir = "./test/"
    Path(test_saving_dir).mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.style.use(args.stylesheet)
    args.mixture_approaches = ["MM"]
    mc_params = MonteCarloRunParameters.from_args(args)
    solver_params = solver_params_from_mc_params(mc_params)
    mc_params.num_configurations = 1
    mc_params.runs_per_configuration = 1
    group: SE2 = None
    state_type: SE2State = None
    if mc_params.dims == 2:
        group = SE2
        state_type = SE2State
    elif mc_params.dims == 3:
        group = SE3
        state_type = SE3State
    mc_params.generate_random_transformations()
    mc_dir = test_saving_dir
    problem_parameter_dir = mc_dir
    opt_result_dir = mc_dir
    fname = os.path.join(test_saving_dir, "mc_params.json")
    mc_params.to_json(fname)
    problem_param_list = load_problem_param_list(
        problem_parameter_dir, create_new=True, mc_params=mc_params
    )

    result_dict = {}
    problem_params = problem_param_list[0]
    T_ts = group.from_components(*(problem_params.get_C_ts_and_r_t_st()))
    true_x: SE2State = state_type(T_ts, stamp=0.0)
    metric_dict = {}

    for mixture_approach in args.mixture_approaches:
        result_dict[mixture_approach] = {}
        result, _ = solve_psr_problem(
            problem_params.ref_landmarks,
            problem_params.source_landmarks,
            problem_params.ref_covs,
            problem_params.source_covs,
            residual_type=mixture_approach,
            solver_params=solver_params,
            verbose=True,
        )

        result_dict[mixture_approach]: OptimizationResult = (
            OptimizationResult.from_batch_navlie_result(
                result,
                true_x,
                "P1",
                problem_params,
            )
        )
        # Save and load the OptimizationResult as a test.
        metric_dict[mixture_approach] = {}
        metric_dict[mixture_approach]["RMSE"] = result_dict[mixture_approach].rmse
        metric_dict[mixture_approach]["Num. Iter."] = result_dict[
            mixture_approach
        ].num_iterations
        metric_dict[mixture_approach]["NEES"] = result_dict[
            mixture_approach
        ].nees.squeeze()

    # Print metrics
    df = pd.DataFrame.from_dict(metric_dict)
    df = df.transpose()
    df = df[["RMSE", "Num. Iter.", "NEES"]]
    print(df)

    if args.dims == 2:
        plot_results_2d(args, mc_params, problem_params, result_dict)
        plt.savefig(
            os.path.join(args.fig_dir, "2d_plot_result_drawn.pdf"), bbox_inches="tight"
        )
        plot_results_2d(args, mc_params, problem_params, {})
        plt.savefig(os.path.join(args.fig_dir, "2d_psr_setup.pdf"), bbox_inches="tight")

    if args.dims == 3:
        plot_results_3d(args, mc_params, problem_params, result_dict)


# plt.show()


def plot_results_3d(args, mc_params, problem_params, result_dict):
    bounds = [b * 1.5 for b in args.monte_carlo_landmark_generation_bounds]
    color_ref = "black"
    color_source = "red"
    if mc_params.dims == 3:
        fig = plt.figure(figsize=(10, 10))
        ax: plt.Axes = fig.add_subplot(projection="3d")
        ax.scatter(
            problem_params.ref_landmarks[0, :],
            problem_params.ref_landmarks[1, :],
            problem_params.ref_landmarks[2, :],
            facecolors=color_ref,
        )
        print(problem_params.C_st.T @ problem_params.C_st)
        ax.scatter(
            problem_params.source_landmarks[0, :],
            problem_params.source_landmarks[1, :],
            problem_params.source_landmarks[2, :],
            facecolors=color_source,
        )
    draw_reference_frame(ax, color_ref, np.array([0, 0, 0]), np.eye(3), 1, None, None)
    draw_reference_frame(
        ax,
        color_source,
        problem_params.r_s_ts,
        problem_params.C_st,
        1.2,
    )
    cmap = sns.color_palette("colorblind")

    for lv1, (mixture_approach, result) in enumerate(result_dict.items()):
        result: OptimizationResult = result
        color_result = cmap[lv1]
        _, _, C_st_hat, r_s_ts_hat = result.decompose_estimated_x()
        draw_reference_frame(ax, color_result, r_s_ts_hat, C_st_hat, 1, None, None)

        source_landmarks_hat = (
            C_st_hat @ problem_params.ref_landmarks + r_s_ts_hat.reshape(-1, 1)
        )
        ax.scatter(
            source_landmarks_hat[0, :],
            source_landmarks_hat[1, :],
            source_landmarks_hat[2, :],
            facecolors="none",
            edgecolors=color_result,
            label=mixture_approach,
        )

    ax.legend()
    plt.show()
    plt.show()


def plot_results_2d(args, mc_params, problem_params: ProblemParameters, result_dict):
    bounds = [b * 1.5 for b in args.monte_carlo_landmark_generation_bounds]
    color_ref = "black"
    color_source = "red"
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    plot_problem_setup(
        ax,
        problem_params.ref_landmarks,
        problem_params.source_landmarks,
        problem_params.ref_covs,
        problem_params.source_covs,
        bounds,
    )
    C_ts, r_s_ts = problem_params.get_C_ts_and_r_t_st()
    draw_reference_frame(ax, color_ref, np.array([0, 0]), np.eye(2), 1, 0.2, 0.3)
    draw_reference_frame(
        ax,
        color_source,
        problem_params.r_s_ts,
        problem_params.C_st,
        1.2,
        0.2,
        0.3,
    )
    # b) Draw result
    cmap = sns.color_palette("colorblind")

    for lv1, (mixture_approach, result) in enumerate(result_dict.items()):
        result: OptimizationResult = result
        color_result = cmap[lv1]
        _, _, C_st_hat, r_s_ts_hat = result.decompose_estimated_x()
        draw_reference_frame(ax, color_result, r_s_ts_hat, C_st_hat, 1, 0.2, 0.3)

        source_landmarks_hat = (
            C_st_hat @ problem_params.ref_landmarks + r_s_ts_hat.reshape(-1, 1)
        )
        ax.scatter(
            source_landmarks_hat[0, :],
            source_landmarks_hat[1, :],
            facecolors="none",
            edgecolors=color_result,
            label=mixture_approach,
        )
    ax.set_xlabel("$x_1$ (m)")
    ax.set_ylabel("$x_2$ (m)")
    ax.legend(loc="upper left")
    plt.show()


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
