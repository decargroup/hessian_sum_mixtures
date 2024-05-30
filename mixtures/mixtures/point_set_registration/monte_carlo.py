from typing import Dict
import json
from typing import Dict
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import dill as pickle
from mixtures.point_set_registration.point_set_registration import (
    corrupt_landmarks_with_noise,
    duplicate_landmarks,
    generate_covariances,
    generate_random_transformation,
    solve_psr_problem,
)
from mixtures.tqdm_joblib import tqdm_joblib
from navlie.lib.states import SE2State, SE3State, State
from navlie.types import StateWithCovariance
from navlie.utils import GaussianResult, GaussianResultList, MonteCarloResult
from pymlg import SE2, SE3


@dataclass
class ProblemParameters:
    problem_id: str
    dims: int
    # Landmarks are arrays
    # of shape (dims, n_landmarks_total)
    ref_landmarks_no_noise: np.ndarray
    source_landmarks_no_noise: np.ndarray
    ref_landmarks: np.ndarray
    source_landmarks: np.ndarray
    ref_covs: List[np.ndarray]
    source_covs: List[np.ndarray]
    C_st: np.ndarray
    r_s_ts: np.ndarray

    def get_C_ts_and_r_t_st(self):
        # this is a bit misleading type hint
        # group can be SE3 or SE2.
        # but methods are pretty much the same.
        group: SE2 = None
        if self.dims == 2:
            group = SE2
        elif self.dims == 3:
            group = SE3

        T_st = group.from_components(self.C_st, self.r_s_ts)
        T_ts = group.inverse(T_st)
        C_ts, r_t_st = group.to_components(T_ts)
        return C_ts, r_t_st

    def print(self, details=False):
        print(f"Problem ID: {self.problem_id}")
        print(f"Dimensions: {self.dims}")
        print(f"Ref landmark shape: {self.ref_landmarks.shape}")
        print(f"Source landmark shape: {self.source_landmarks.shape}")
        print(f"Ref landmark no noise shape: {self.ref_landmarks_no_noise.shape}")
        print(f"Source landmark no noise shape: {self.source_landmarks_no_noise.shape}")
        print("C_st")
        print(self.C_st)
        print("r_s_ts")
        print(self.r_s_ts)
        if details:
            print(f"Reference landmarks: {self.ref_landmarks}")
            print(f"Source landmarks: {self.source_landmarks}")
            print(f"Reference covariances: {self.ref_covs}")
            print(f"Source covariances: {self.source_covs}")

    def to_data_dict(self):
        data = {
            "problem_id": self.problem_id,
            "dims": self.dims,
            "ref_landmarks_no_noise": self.ref_landmarks_no_noise.reshape(-1, 1)
            .squeeze()
            .tolist(),
            "ref_landmarks": self.ref_landmarks.reshape(-1, 1).squeeze().tolist(),
            "source_landmarks_no_noise": self.source_landmarks_no_noise.reshape(-1, 1)
            .squeeze()
            .tolist(),
            "source_landmarks": self.source_landmarks.reshape(-1, 1).squeeze().tolist(),
            "ref_covs": [
                cov.reshape(-1, 1).squeeze().tolist() for cov in self.ref_covs
            ],
            "source_covs": [
                cov.reshape(-1, 1).squeeze().tolist() for cov in self.source_covs
            ],
            "C_st": self.C_st.reshape(-1, 1).squeeze().tolist(),
            "r_s_ts": self.r_s_ts.reshape(-1, 1).squeeze().tolist(),
        }
        return data

    @staticmethod
    def from_data_dict(data):
        return ProblemParameters(
            problem_id=data["problem_id"],
            dims=data["dims"],
            ref_landmarks_no_noise=np.array(data["ref_landmarks_no_noise"]).reshape(
                data["dims"], -1
            ),
            source_landmarks_no_noise=np.array(
                data["source_landmarks_no_noise"]
            ).reshape(data["dims"], -1),
            ref_landmarks=np.array(data["ref_landmarks"]).reshape(data["dims"], -1),
            source_landmarks=np.array(data["source_landmarks"]).reshape(
                data["dims"], -1
            ),
            ref_covs=[
                np.array(ref_cov_data).reshape(data["dims"], data["dims"])
                for ref_cov_data in data["ref_covs"]
            ],
            source_covs=[
                np.array(source_cov_data).reshape(data["dims"], data["dims"])
                for source_cov_data in data["source_covs"]
            ],
            C_st=np.array(data["C_st"]).reshape(data["dims"], data["dims"]),
            r_s_ts=np.array(data["r_s_ts"]),
        )

    def to_json(self, fname: str):
        data = self.to_data_dict()
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    @staticmethod
    def from_json(fname: str):
        with open(fname, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ProblemParameters.from_data_dict(data)


@dataclass
class MonteCarloRunParameters:
    run_id: str
    dims: int
    landmark_generation_bounds: List[float]
    runs_per_configuration: int
    num_configurations: int
    transformation_T_st_list: List[np.ndarray]
    angle_range: List[float]
    r_range: List[float]
    cluster_size: int
    cluster_spread: float
    fraction_of_landmarks_to_cluster: float
    num_landmarks: int  # Before duplication.
    ref_noise_stddev_bounds: List[float]
    meas_noise_stddev_bounds: List[float]
    solver: str
    step_tol: float
    gradient_tol: float
    ftol: float
    convergence_criterion: str
    max_iters: int
    initial_normalization_constant_hsm: float
    tau: float

    def __str__(self):
        mc_string = f"Monte Carlo Run With Parameters:\n\
Run ID: {self.run_id}\n\
Dimensions: {self.dims}\n\
Landmark generation bounds: {self.landmark_generation_bounds}\n\
Runs per configuration: {self.runs_per_configuration}\n\
Number of configurations: {self.num_configurations}\n\
Angle range: {self.angle_range}\n\
r range: {self.r_range}\n\
Cluster size: {self.cluster_size}\n\
Cluster spread: {self.cluster_spread}\n\
Fraction of landmarks to cluster: {self.fraction_of_landmarks_to_cluster}\n\
Number of landmarks: {self.num_landmarks}\n\
Reference noise stddev bounds: {self.ref_noise_stddev_bounds}\n\
Measurement noise stddev bounds: {self.meas_noise_stddev_bounds}\n\
Solver: {self.solver}\n\
Step tolerance: {self.step_tol}\n\
Gradient tolerance: {self.gradient_tol}\n\
Maximum iterations: {self.max_iters}\n\
LM tau: {self.tau}\n\
Initial normalization constant HSM: {self.initial_normalization_constant_hsm}\n\
ftol: {self.ftol}\n"
        return mc_string

    @staticmethod
    def from_args(args):
        return MonteCarloRunParameters(
            run_id=args.monte_carlo_run_id,
            dims=args.dims,
            landmark_generation_bounds=args.monte_carlo_landmark_generation_bounds,
            runs_per_configuration=args.runs_per_configuration,
            num_configurations=args.num_configurations,
            transformation_T_st_list=[],
            angle_range=args.monte_carlo_transformation_angle_range,
            r_range=args.monte_carlo_transformation_r_range,
            cluster_size=args.cluster_size,
            cluster_spread=args.cluster_spread,
            fraction_of_landmarks_to_cluster=args.fraction_of_landmarks_to_cluster,
            num_landmarks=args.num_landmarks,
            ref_noise_stddev_bounds=args.ref_noise_stddev_bounds,
            meas_noise_stddev_bounds=args.meas_noise_stddev_bounds,
            solver=args.solver,
            step_tol=args.step_tol,
            gradient_tol=args.gradient_tol,
            ftol=args.ftol,
            convergence_criterion=args.convergence_criterion,
            max_iters=args.max_iters,
            tau=args.tau,
            initial_normalization_constant_hsm=args.initial_normalization_constant_hsm,
        )

    def generate_random_transformations(self):
        self.transformation_T_st_list = []
        for lv1 in range(self.runs_per_configuration):
            C_st, r_s_ts, _, _ = generate_random_transformation(
                angle_range=self.angle_range, r_range=self.r_range, dims=self.dims
            )
            group: SE2 = None
            if self.dims == 2:
                group = SE2
            elif self.dims == 3:
                group = SE3

            self.transformation_T_st_list.append(group.from_components(C_st, r_s_ts))

    def to_json(self, fname: str):
        data = {
            "run_id": self.run_id,
            "dims": self.dims,
            "landmark_generation_bounds": self.landmark_generation_bounds,
            "runs_per_configuration": self.runs_per_configuration,
            "num_configurations": self.num_configurations,
            "transformation_T_st_list": [
                T.reshape(-1, 1).squeeze().tolist()
                for T in self.transformation_T_st_list
            ],
            "angle_range": self.angle_range,
            "r_range": self.r_range,
            "cluster_size": self.cluster_size,
            "cluster_spread": self.cluster_spread,
            "fraction_of_landmarks_to_cluster": self.fraction_of_landmarks_to_cluster,
            "num_landmarks": self.num_landmarks,
            "ref_noise_stddev_bounds": self.ref_noise_stddev_bounds,
            "meas_noise_stddev_bounds": self.meas_noise_stddev_bounds,
            "solver": self.solver,
            "step_tol": self.step_tol,
            "gradient_tol": self.gradient_tol,
            "ftol": self.ftol,
        }
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def from_json(fname: str):
        with open(fname, "r", encoding="utf-8") as f:
            data = json.load(f)
        return MonteCarloRunParameters(
            run_id=data["run_id"],
            dims=data["dims"],
            landmark_generation_bounds=data["landmark_generation_bounds"],
            runs_per_configuration=data["runs_per_configuration"],
            num_configurations=data["num_configurations"],
            transformation_T_st_list=[
                np.array(T).reshape(data["dims"] + 1, data["dims"] + 1)
                for T in data["transformation_T_st_list"]
            ],
            angle_range=data["angle_range"],
            r_range=data["r_range"],
            cluster_size=data["cluster_size"],
            cluster_spread=data["cluster_spread"],
            fraction_of_landmarks_to_cluster=data["fraction_of_landmarks_to_cluster"],
            num_landmarks=data["num_landmarks"],
            ref_noise_stddev_bounds=data["ref_noise_stddev_bounds"],
            meas_noise_stddev_bounds=data["meas_noise_stddev_bounds"],
            solver=data["solver"],
            step_tol=data["step_tol"],
            gradient_tol=data["gradient_tol"],
            ftol=data["ftol"],
        )


class OptimizationResult:
    """
    Still needs num_iterations, convergence success, rmse.
    """

    problem_id: str
    problem_params: ProblemParameters
    distances_to_optimum: List[float]
    deltas_to_optimum: List[np.ndarray]
    cost_history: List[float]
    true_x: State
    rmse: float
    num_iterations: int
    nees: float
    anees: float
    convergence_success: bool
    error: np.ndarray
    info_matrix: np.ndarray
    covariance: np.ndarray

    def __init__(
        self,
        problem_id: str,
        problem_params: ProblemParameters,
        distances_to_optimum: List[float],
        deltas_to_optimum: List[np.ndarray],
        cost_history: List[float],
        true_x: State,
        est_x: State,
        info_matrix: np.ndarray,
        total_time: float = None,
    ):
        self.problem_id = problem_id
        self.problem_params = problem_params
        self.distances_to_optimum = distances_to_optimum
        self.deltas_to_optimum = deltas_to_optimum
        self.cost_history = cost_history
        self.true_x = true_x
        self.est_x = est_x
        self.info_matrix = info_matrix
        self.total_time = total_time
        self.compute_metrics()

    def to_pickle(self, fname: str):
        with open(fname, "wb") as f:
            pickle.dump(self, f)

    def from_pickle(fname: str):
        with open(fname, "rb") as f:
            return pickle.load(f)

    def decompose_estimated_x(self):
        group: SE2 = self.est_x.group
        C_ts_hat, r_t_st_hat = group.to_components(self.est_x.value)
        C_st_hat, r_s_ts_hat = group.to_components(np.linalg.inv(self.est_x.value))
        return C_ts_hat, r_t_st_hat, C_st_hat, r_s_ts_hat

    def compute_metrics(self, convergence_threshold=0.01):
        self.rmse = self.distances_to_optimum[-1]
        self.num_iterations = (
            len(self.cost_history) - 1
        )  # First one is cost at zero'th iteration
        self.convergence_success = self.rmse < convergence_threshold
        error = self.deltas_to_optimum[-1].reshape(-1, 1)
        self.error = error
        self.nees = (error.T @ self.info_matrix @ error).squeeze()
        self.anees = self.nees / self.true_x.dof
        self.covariance = np.linalg.inv(self.info_matrix)

    @staticmethod
    def from_batch_navlie_result(
        opt_nv_res,
        true_x: State,
        problem_id: str,
        problem_params: ProblemParameters,
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
        x_hat: State = opt_nv_res["variables"][key]

        distances_to_optimum = np.array(
            [np.linalg.norm(true_x.minus(x)) for x in x_history]
        ).tolist()
        total_time = opt_nv_res["summary"].time

        return OptimizationResult(
            problem_id=problem_id,
            problem_params=problem_params,
            distances_to_optimum=distances_to_optimum,
            deltas_to_optimum=deltas_to_optimum,
            cost_history=opt_nv_res["summary"].entire_cost.tolist(),
            true_x=true_x,
            est_x=x_hat,
            info_matrix=info_matrix,
            total_time=total_time,
        )


def load_problem_param_list(
    problem_parameter_dir: str, create_new: bool = False, mc_params=None
) -> List[ProblemParameters]:
    if create_new:
        print("Generating new problem parameters.. ")
        problem_param_list = get_problem_parameter_list_from_mc_params(mc_params)

        print("Saving new problem parameters.. ")
        for problem_params in tqdm(problem_param_list):
            problem_params.to_json(
                fname=os.path.join(
                    problem_parameter_dir,
                    f"{problem_params.problem_id}.json",
                )
            )
    else:
        print(f"Loading problem parameters..")
        problem_param_list = []
        for fname in tqdm(list(Path(problem_parameter_dir).glob("*.json"))):
            problem_param_list.append(ProblemParameters.from_json(fname))
    return problem_param_list


def get_problem_parameter_list_from_mc_params(
    mc_params: MonteCarloRunParameters,
) -> List[ProblemParameters]:
    # this is a bit misleading type hint
    # group can be SE3 or SE2.
    # but methods are pretty much the same.
    group: SE2 = None
    if mc_params.dims == 2:
        group = SE2
    elif mc_params.dims == 3:
        group = SE3

    problem_param_list = []
    for lv1 in tqdm(range(mc_params.num_configurations)):
        # Generate landmarks
        ref_landmarks_no_noise = np.random.uniform(
            mc_params.landmark_generation_bounds[0],
            mc_params.landmark_generation_bounds[1],
            size=(mc_params.dims, mc_params.num_landmarks),
        )
        ref_landmarks_no_noise = duplicate_landmarks(
            ref_landmarks_no_noise,
            mc_params.cluster_size,
            mc_params.cluster_spread,
            mc_params.fraction_of_landmarks_to_cluster,
        )
        for lv2, T_st in enumerate(mc_params.transformation_T_st_list):
            C_st, r_s_ts = group.to_components(T_st)
            ref_covs = generate_covariances(
                mc_params.ref_noise_stddev_bounds[0],
                mc_params.ref_noise_stddev_bounds[1],
                ref_landmarks_no_noise.shape[1],
                mc_params.dims,
            )
            source_covs = generate_covariances(
                mc_params.meas_noise_stddev_bounds[0],
                mc_params.meas_noise_stddev_bounds[1],
                ref_landmarks_no_noise.shape[1],
                mc_params.dims,
            )
            source_landmarks_no_noise = C_st @ ref_landmarks_no_noise + r_s_ts.reshape(
                -1, 1
            )
            ref_landmarks = corrupt_landmarks_with_noise(
                ref_landmarks_no_noise, ref_covs
            )
            source_landmarks = corrupt_landmarks_with_noise(
                source_landmarks_no_noise, source_covs
            )
            problem_params = ProblemParameters(
                f"L{lv1}T{lv2}",  # landmark config lv1, transformation idx lv2.
                mc_params.dims,
                ref_landmarks_no_noise,
                source_landmarks_no_noise,
                ref_landmarks,
                source_landmarks,
                ref_covs,
                source_covs,
                C_st,
                r_s_ts,
            )
            problem_param_list.append(problem_params)
    return problem_param_list


# There's a good chunk of code duplication between the postprocessing for vanilla example and psr, not amazing
# TODO: Fix
def metric_dataframes(
    mixture_approaches: List[str],
    opt_result_dir: str,
    read_from_csv=False,
    csv_folder=None,
):
    metric_dict = {}
    metric_names = ["RMSE (deg)", "RMSE (m)", "NEES", "ANEES", "Avg Iter.", "Time (s)"]
    if not read_from_csv:
        for metric in metric_names:
            metric_dict[metric] = {}
        for mixture_approach in mixture_approaches:
            list_opt_result = list(
                Path(os.path.join(opt_result_dir, mixture_approach)).glob("*.pkl")
            )
            # list_opt_result = list_opt_result[0:10]
            opt_result_list: List[OptimizationResult] = []
            print(f"Loading files for {mixture_approach}")
            for fname in tqdm(list_opt_result):
                opt_result_list.append(OptimizationResult.from_pickle(fname))
            print(f"Computing metrics..")
            for opt_result in tqdm(opt_result_list):
                opt_result.compute_metrics()

            x_hat_list = [
                StateWithCovariance(opt_result.est_x, opt_result.covariance)
                for opt_result in opt_result_list
            ]
            true_x_list = [opt_result.true_x for opt_result in opt_result_list]
            list_of_gaussian_result_list = [
                GaussianResultList([GaussianResult(x_hat, true_x)])
                for x_hat, true_x in zip(x_hat_list, true_x_list)
            ]
            mc_result = MonteCarloResult(list_of_gaussian_result_list)
            #     # Save and load the OptimizationResult as a test.
            metric_dict[mixture_approach] = {}
            # Have to consider the RMSEs separately.
            rmse_rad = np.sqrt(
                np.array([t.error.squeeze()[0] for t in mc_result.trial_results]) ** 2,
            )
            rmse_deg = rmse_rad * 180 / np.pi

            rmse_m = np.array(
                [
                    np.linalg.norm(t.error.squeeze()[1:], 2) ** 2
                    for t in mc_result.trial_results
                ]
            )

            nees = [opt_result.nees for opt_result in opt_result_list]
            anees = [opt_result.anees for opt_result in opt_result_list]
            metric_dict["RMSE (deg)"][mixture_approach] = rmse_deg
            metric_dict["RMSE (m)"][mixture_approach] = rmse_m
            metric_dict["NEES"][mixture_approach] = nees
            metric_dict["ANEES"][mixture_approach] = anees
            metric_dict["Avg Iter."][mixture_approach] = np.array(
                [opt_result.num_iterations for opt_result in opt_result_list]
            )
            metric_dict["Time (s)"][mixture_approach] = np.array(
                [opt_result.total_time for opt_result in opt_result_list]
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


def average_metric_table(metric_dict: Dict[str, pd.DataFrame], print_table=True):
    """_summary_

    Parameters
    ----------
    metric_dict : Dict[str, pd.DataFrame]
        metric_dict[metric_name] contains a dataframe with
        columns corresponding to the mixtures and rows corresponding to the runs.
    """
    metric_names = ["RMSE (deg)", "RMSE (m)", "NEES", "ANEES", "Avg Iter.", "Time (s)"]

    approaches = metric_dict["RMSE (deg)"].columns
    average_metrics_dict = {}
    for approach in approaches:
        average_metrics_dict[approach] = {}
        for metric_name in metric_names:
            average_metrics_dict[approach][metric_name] = np.mean(
                np.array([metric_dict[metric_name][approach]])
            )
    df = pd.DataFrame.from_dict(average_metrics_dict)
    df = df.transpose()
    df_all = df[["RMSE (deg)", "RMSE (m)", "ANEES", "NEES", "Avg Iter.", "Time (s)"]]
    df_all = df.rename(
        columns={
            "Avg Iter.": "Avg Iterations",
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


def format_df(df):
    df_styled = df.style.highlight_min(
        subset=["RMSE (deg)", "RMSE (m)", "NEES", "ANEES", "Avg Iterations"],
        axis=0,
        props="textbf:--rwrap;",
    )

    df_styled = df_styled.format(
        {
            "RMSE (deg)": "{:,.2e}".format,
            "RMSE (m)": "{:,.2e}".format,
            "NEES": "{:,.2f}".format,
            "ANEES": "{:,.2f}".format,
            "Avg Iterations": "{:,.2f}".format,
        }
    )
    return df_styled


def optimize_trials(
    mixture_approaches: List[str],
    opt_result_dir: str,
    problem_param_list: List[ProblemParameters],
    n_jobs=1,
    verbose=False,
    solver_params: Dict = None,
):
    print(f"Starting Monte Carlo for {opt_result_dir}")
    for mixture_approach in mixture_approaches:
        Path(os.path.join(opt_result_dir, mixture_approach)).mkdir(
            exist_ok=True, parents=True
        )
    if problem_param_list[0].C_st.shape[0] == 2:
        group = SE2
        state_type = SE2State
    elif problem_param_list[0].C_st.shape[0] == 3:
        group = SE3
        state_type = SE3State

    def trial(problem_params: ProblemParameters):
        # print(f"Running problem {problem_params.problem_id}")
        for mixture_approach in mixture_approaches:
            C_ts, r_t_st = problem_params.get_C_ts_and_r_t_st()
            T_ts = group.from_components(C_ts, r_t_st)
            true_x: SE2State = state_type(T_ts, stamp=0.0)

            result, _ = solve_psr_problem(
                problem_params.ref_landmarks,
                problem_params.source_landmarks,
                problem_params.ref_covs,
                problem_params.source_covs,
                residual_type=mixture_approach,
                solver_params=solver_params,
                verbose=verbose,
            )
            opt_result: OptimizationResult = (
                OptimizationResult.from_batch_navlie_result(
                    result,
                    true_x,
                    problem_params.problem_id,
                    problem_params,
                )
            )
            opt_result.to_pickle(
                os.path.join(
                    opt_result_dir,
                    mixture_approach,
                    f"{problem_params.problem_id}.pkl",
                )
            )

    if n_jobs > 1 or n_jobs == -1:
        with tqdm_joblib(
            tqdm(desc=f"Monte Carlo", total=len(problem_param_list))
        ) as progress_bar:
            Parallel(n_jobs=int(n_jobs))(
                delayed(trial)(problem_params) for problem_params in problem_param_list
            )
    else:
        for problem_params in problem_param_list:
            trial(problem_params)
