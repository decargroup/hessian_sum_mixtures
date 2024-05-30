import json
import os
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import dill as pickle
import numpy as np
from joblib import Parallel, delayed
from scipy import optimize  # For true x computation
from scipy.optimize import OptimizeResult
from scipy.stats import multivariate_normal
from tqdm import tqdm

from mixtures.gaussian_mixtures import HessianSumMixtureResidualDirectHessian
from mixtures.solver import ProblemExtended, solve_vector_mixture_problem
from mixtures.vanilla_mixture.mixture_utils import (
    create_residuals,
    get_component_residuals,
)
from mixtures.vanilla_mixture.plotting import get_plot_bounds
from navlie.batch.gaussian_mixtures import (
    HessianSumMixtureResidual as HessianSumMixtureResidualStandardCompatibility,
)
from navlie.batch.gaussian_mixtures import (
    GaussianMixtureResidual,
    MaxMixtureResidual,
    MaxSumMixtureResidual,
    SumMixtureResidual,
)
from navlie.lib.states import State, VectorState


@dataclass
class GaussianMixtureParameters:
    mixture_id: str
    weights: List[float]
    means: List[np.ndarray]
    covariances: List[np.ndarray]

    def print(self):
        print(f"Mixture ID: {self.mixture_id}")
        print("Weights")
        print(self.weights)
        print("Means")
        for mean in self.means:
            print(mean.reshape(1, -1))
        print("Cov diags")
        for cov in self.covariances:
            print(cov.diagonal())


VanillaMixtureMetric = namedtuple(
    "VanillaMixtureMetric",
    [
        "rmse",
        "num_iterations",
        "convergence_success",
        "nees",
        "dof",
        "distance_to_optimum",
    ],
)


class OptimizationResult:
    """
    Still needs num_iterations, convergence success, rmse.
    """

    mix_id: str
    gaussian_mix_params: GaussianMixtureParameters
    distances_to_optimum: List[float]
    deltas_to_optimum: List[np.ndarray]
    cost_history: List[float]
    true_x: State
    rmse: float
    num_iterations: int
    nees: float
    convergence_success: bool
    x0: np.ndarray
    info_matrix: np.ndarray

    def __init__(
        self,
        mix_id: str,
        gaussian_mix_params: GaussianMixtureParameters,
        distances_to_optimum: List[float],
        deltas_to_optimum: List[np.ndarray],
        cost_history: List[float],
        true_x: State,
        x0: np.ndarray,
        info_matrix: np.ndarray,
        est_x: State,
        x_history: List[State],
    ):
        self.mix_id = mix_id
        self.gaussian_mix_params = gaussian_mix_params
        self.distances_to_optimum = distances_to_optimum
        self.deltas_to_optimum = deltas_to_optimum
        self.cost_history = cost_history
        self.true_x = true_x
        self.x0 = x0
        self.info_matrix = info_matrix
        self.est_x = est_x
        self.x_history = x_history

    def compute_metrics(self, convergence_threshold=0.01) -> VanillaMixtureMetric:
        self.rmse = np.sqrt(np.sum((self.true_x.minus(self.est_x)).squeeze() ** 2))
        self.num_iterations = (
            len(self.cost_history) - 1
        )  # First one is cost at iteration 0
        self.convergence_success = self.rmse < convergence_threshold
        error = self.deltas_to_optimum[-1].reshape(-1, 1)
        self.nees = error.T @ self.info_matrix @ error
        self.nees = self.nees.squeeze()

        return VanillaMixtureMetric(
            self.rmse,
            self.num_iterations,
            self.convergence_success,
            self.nees,
            self.true_x.dof,
            self.distances_to_optimum[-1],
        )

    def to_pickle(self, fname: str):
        with open(fname, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def from_pickle(fname: str):
        with open(fname, "rb") as f:
            return pickle.load(f)

    def print(self):
        self.gaussian_mix_params.print()
        print("x0")
        print(self.x0)

    @staticmethod
    def from_batch_navlie_result(
        opt_nv_res,
        true_x: State,
        mix_id: str,
        gaussian_mix_params: GaussianMixtureParameters,
        x0: State,
    ):
        """
        Create our OptimizationResult from the dictionary opt_nv_res
        that is
        output by problem.solve() in navlie.
        The dict has to have keys "variables", "summary", "info_matrix"
        """
        info_matrix = opt_nv_res["info_matrix"]
        iterate_history = opt_nv_res["summary"].iterate_history
        # x_history = [
        # iterate_history[i][0]["x"].value for i in range(len(iterate_history))
        # ]
        key = list(iterate_history[0][0].keys())[0]
        x_history: List[State] = [
            iterate_history[i][0][key] for i in range(len(iterate_history))
        ]
        deltas_to_optimum = [x.minus(true_x) for x in x_history]
        est_x: State = opt_nv_res["variables"][key]

        distances_to_optimum = np.array(
            [np.linalg.norm(true_x.minus(x)) for x in x_history]
        ).tolist()

        return OptimizationResult(
            mix_id,
            gaussian_mix_params,
            distances_to_optimum,
            deltas_to_optimum,
            opt_nv_res["summary"].entire_cost,
            true_x,
            x0,
            info_matrix,
            est_x,
            x_history,
        )


@dataclass
class MonteCarloRunParameters:
    monte_carlo_run_id: str
    dims: int
    n_components: int
    mean_ranges: List[np.ndarray]
    stddev_ranges: List[np.ndarray]
    component_multiplier_ranges: List[np.ndarray]
    initial_position_ranges: List[float]  # 2 numbers
    first_component_weight_range: List[float]
    method_to_create_smaller_components: str
    num_initial_position_per_mixture: int
    method_initial_position_choice: str
    num_mixtures: int
    use_triggs_hsm: bool
    ceres_triggs_patch: bool
    solver: str
    max_iters: int
    step_tol: float
    gradient_tol: float
    ftol: float
    tau: float  # LM parameter
    convergence_criterion: str  # One of "step", "rel_cost", "gradient"
    initial_normalization_constant_hsm: float

    @staticmethod
    def from_args(args):
        # Massage arguments to initialize parameters of interest
        args.dims = args.dims
        n_components = int(
            len(args.mean_ranges) / 2
        )  # There is an upper and lower bound to eachn component.

        mean_ranges = [
            np.array(args.mean_ranges[lv1 * 2 : (1 + lv1) * 2])
            for lv1 in range(n_components)
        ]

        stddev_ranges = [
            np.array(args.stddev_ranges[lv1 * 2 : (1 + lv1) * 2])
            for lv1 in range(n_components)
        ]

        component_multiplier_ranges = [
            np.array(args.component_multiplier_ranges[lv1 * 2 : (1 + lv1) * 2])
            for lv1 in range(n_components)
        ]
        initial_position_ranges = [
            np.array(args.initial_position_ranges[lv1 * 2 : (1 + lv1) * 2])
            for lv1 in range(args.dims)
        ]

        return MonteCarloRunParameters(
            monte_carlo_run_id=args.monte_carlo_run_id,
            dims=args.dims,
            n_components=n_components,
            mean_ranges=mean_ranges,
            stddev_ranges=stddev_ranges,
            component_multiplier_ranges=component_multiplier_ranges,
            initial_position_ranges=initial_position_ranges,
            first_component_weight_range=args.first_component_weight_range,
            method_to_create_smaller_components=args.method_to_create_smaller_components,
            num_initial_position_per_mixture=int(args.num_initial_position_per_mixture),
            method_initial_position_choice=args.method_initial_position_choice,
            num_mixtures=args.num_mixtures,
            use_triggs_hsm=args.use_triggs_hsm,
            ceres_triggs_patch=args.ceres_triggs_patch,
            solver=args.solver,
            max_iters=args.max_iters,
            step_tol=args.step_tol,
            gradient_tol=args.gradient_tol,
            ftol=args.ftol,
            tau=args.tau,
            convergence_criterion=args.convergence_criterion,
            initial_normalization_constant_hsm=args.initial_normalization_constant_hsm,
        )

    def to_pickle(self, fname: str):
        with open(fname, "wb") as f:
            pickle.dump(self, f)

    def from_pickle(fname: str):
        with open(fname, "rb") as f:
            return pickle.load(f)


def setup_optimization_runs(
    mc_params: MonteCarloRunParameters,
) -> Tuple[List[GaussianMixtureParameters], List[np.ndarray]]:
    # Generate many mixtures
    gaussian_mixture_list: List[GaussianMixtureParameters] = []
    for lv1 in range(mc_params.num_mixtures):
        weights = []
        means = []
        covariances = []
        # Set up weights
        weights.append(
            np.random.uniform(
                low=mc_params.first_component_weight_range[0],
                high=mc_params.first_component_weight_range[1],
            ),
        )
        for lv2 in range(1, mc_params.n_components):
            weights.append((1 - weights[0]) / (mc_params.n_components - 1))
        # Set up means/covariances of components
        for lv2 in range(mc_params.n_components):
            mean = np.random.uniform(
                size=(mc_params.dims, 1),
                low=mc_params.mean_ranges[lv2][0],
                high=mc_params.mean_ranges[lv2][1],
            )
            if (
                lv2 > 0
                and mc_params.method_to_create_smaller_components == "multiplied"
            ):
                multiplier = np.random.uniform(
                    low=mc_params.component_multiplier_ranges[lv2][0],
                    high=mc_params.component_multiplier_ranges[lv2][1],
                )
                cov = multiplier * covariances[0]
            elif (
                lv2 == 0 or mc_params.method_to_create_smaller_components == "specified"
            ):
                stddev = np.random.uniform(
                    low=mc_params.stddev_ranges[lv2][0],
                    high=mc_params.stddev_ranges[lv2][1],
                )

                cov = stddev**2 * np.eye(mc_params.dims)
            else:
                raise Exception("Covariance specificaiton for mixtures is broken. ")
            means.append(mean)
            covariances.append(cov)
        mix_id = str(lv1)
        gaussian_mixture_list.append(
            GaussianMixtureParameters(mix_id, weights, means, covariances)
        )

    if mc_params.method_initial_position_choice == "random":
        for lv1 in range(mc_params.dims):
            cur_initial_positions = np.random.uniform(
                size=(1, mc_params.num_initial_position_per_mixture),
                low=mc_params.initial_position_ranges[lv1][0],
                high=mc_params.initial_position_ranges[lv1][1],
            )
            if lv1 == 0:
                initial_positions = cur_initial_positions
            else:
                initial_positions = np.vstack(
                    [initial_positions, cur_initial_positions]
                )

    if mc_params.method_initial_position_choice == "grid":
        if mc_params.dims == 1:
            initial_positions = np.linspace(
                mc_params.initial_position_ranges[0][0],
                mc_params.initial_position_ranges[0][1],
                num=mc_params.num_initial_position_per_mixture,
            ).tolist()
            initial_positions = [np.array(x) for x in initial_positions]
        if mc_params.dims == 2:
            linspaces = []
            for lv1 in range(mc_params.dims):
                linspaces.append(
                    np.linspace(
                        mc_params.initial_position_ranges[lv1][0],
                        mc_params.initial_position_ranges[lv1][1],
                        num=int(
                            np.ceil(np.sqrt(mc_params.num_initial_position_per_mixture))
                        ),
                    )
                )
            initial_positions = np.vstack(list(map(np.ravel, np.meshgrid(*linspaces))))
            initial_positions.reshape(2, -1).T
            initial_positions = [
                initial_positions[:, lv1] for lv1 in range(initial_positions.shape[1])
            ]
    return gaussian_mixture_list, initial_positions


def get_true_x(gm_params: GaussianMixtureParameters, initial_grid_dx=0.1) -> np.ndarray:
    # Use the mixture parameters to get the true x
    weights = gm_params.weights
    means = [m.squeeze() for m in gm_params.means]
    covariances = gm_params.covariances
    xmin, xmax, ymin, ymax = get_plot_bounds(gm_params.means, gm_params.covariances)
    dims = covariances[0].shape[0]
    nx = int(np.ceil((xmax - xmin) / initial_grid_dx))
    if dims == 1:
        x = np.linspace(xmin, xmax, nx)
        mix_list = []
        unweighted_mix_list = []
        for weight, mean, cov in zip(weights, means, covariances):
            Z = np.zeros(x.shape)
            for i in range(x.shape[0]):
                Z[i] = multivariate_normal.pdf(x[i], mean=mean, cov=cov)
            mix_list.append(weight * Z)
            unweighted_mix_list.append(Z)
        mix = np.sum(np.array(mix_list), axis=0)
        argmax_1d = np.argmax(mix)
        x0_true = np.array([x[argmax_1d]]).reshape(-1)
    if dims == 2:
        ny = int(np.ceil((ymax - ymin) / initial_grid_dx))
        mix_list = []
        unweighted_mix_list = []
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(x, y)  # grid of point

        for weight, mean, cov in zip(weights, means, covariances):
            Z = np.zeros(X.shape)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = multivariate_normal.pdf(
                        np.array([X[i, j], Y[i, j]]), mean=mean, cov=cov
                    )
            mix_list.append(weight * np.expand_dims(Z, axis=2))
            unweighted_mix_list.append(Z)

        mix = np.sum(np.concatenate(mix_list, axis=2), axis=2)
        argmax_2d = np.unravel_index(np.argmax(mix, axis=None), mix.shape)
        x0_true = np.array([X[argmax_2d], Y[argmax_2d]]).reshape(-1)

    res_dict = create_residuals(
        gm_params.weights,
        means,
        covariances,
    )
    msm_res: MaxSumMixtureResidual = res_dict["MSM"]

    def f(x: np.ndarray):
        return np.sum(msm_res.evaluate([VectorState(x)], None) ** 2)

    def jac_f(x: np.ndarray):
        e, H_list = msm_res.evaluate([VectorState(x)], [True])
        return H_list[0].T @ e

    res_true: OptimizeResult = optimize.minimize(
        f, x0_true, jac=jac_f, method="L-BFGS-B"
    )
    x_true = res_true.x

    return x_true


def run_monte_carlo(
    mc_params: MonteCarloRunParameters,
    mixture_approaches: List[str],
    STATE_KEY: str,
    top_result_dir: str,
    n_jobs: int,
):

    mc_result_folder = os.path.join(top_result_dir, mc_params.monte_carlo_run_id)
    fname = os.path.join(
        mc_result_folder,
        f"mc_params.pkl",
    )
    mc_params.to_pickle(fname)
    gaussian_mixture_list, initial_positions = setup_optimization_runs(mc_params)

    STATE_KEY = "x"
    # Also have to save optimization parameters to json.
    for lv1, gm_params in tqdm(
        enumerate(gaussian_mixture_list), total=len(gaussian_mixture_list)
    ):
        component_residuals = get_component_residuals(
            gm_params.means,
            gm_params.covariances,
        )
        weights = gm_params.weights

        residual_dict = {
            "MM": MaxMixtureResidual(component_residuals, weights),
            "SM": SumMixtureResidual(component_residuals, weights),
            "MSM": MaxSumMixtureResidual(component_residuals, weights, 10),
            "HSM": HessianSumMixtureResidualDirectHessian(
                component_residuals,
                weights,
                False,
                False,
            ),
            "HSM_STD": HessianSumMixtureResidualStandardCompatibility(
                component_residuals, weights, no_use_complex_numbers=False
            ),
            "HSM_STD_NO_COMPLEX": HessianSumMixtureResidualStandardCompatibility(
                component_residuals,
                weights,
                no_use_complex_numbers=True,
                normalization_constant=mc_params.initial_normalization_constant_hsm,
            ),
        }
        gm_params: GaussianMixtureParameters = gm_params
        mix_id = str(lv1)

        true_x = get_true_x(gm_params, initial_grid_dx=100)

        def trial(lv2, x0):
            # Have to have another loop for all the optimization start points.
            for approach in mixture_approaches:
                gm_resid = residual_dict[approach]

                # TODO: Move solver params and their creation somewhere else.
                solver_params = {
                    "solver": mc_params.solver,
                    "max_iters": mc_params.max_iters,
                    "step_tol": (
                        mc_params.step_tol
                        if mc_params.convergence_criterion == "step"
                        else None
                    ),
                    "gradient_tol": (
                        mc_params.gradient_tol
                        if mc_params.convergence_criterion == "gradient"
                        else None
                    ),
                    "ftol": (
                        mc_params.ftol
                        if mc_params.convergence_criterion == "rel_cost"
                        else None
                    ),
                    "tau": mc_params.tau,
                }

                x = VectorState(x0, 0.0, STATE_KEY)
                problem = ProblemExtended(
                    solver=solver_params["solver"],
                    max_iters=solver_params["max_iters"],
                    step_tol=solver_params["step_tol"],
                    gradient_tol=solver_params["gradient_tol"],
                    ftol=solver_params["ftol"],
                    tau=solver_params["tau"],
                    verbose=False,
                )
                problem.add_residual(gm_resid)
                problem.add_variable(STATE_KEY, x)
                opt_nv_res = problem.solve()
                opt_result = OptimizationResult.from_batch_navlie_result(
                    opt_nv_res,
                    VectorState(true_x, 0.0, STATE_KEY),
                    mix_id,
                    gm_params,
                    x0,
                )

                res_dir = os.path.join(
                    top_result_dir, mc_params.monte_carlo_run_id, approach
                )
                Path(res_dir).mkdir(parents=True, exist_ok=True)

                fname = os.path.join(
                    res_dir,
                    f"opt_result_mix_id_{gm_params.mixture_id}_x0_num_{lv2}.pkl",
                )
                opt_result.to_pickle(fname)

        if n_jobs > 1 or n_jobs == -1:
            Parallel(n_jobs=n_jobs)(
                delayed(trial)(lv2, x0) for lv2, x0 in enumerate(initial_positions)
            )
        else:  # Easier to debug..
            for lv2, x0 in enumerate(initial_positions):
                trial(lv2, x0)
