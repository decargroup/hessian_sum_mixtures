import numpy as np
from typing import List
import random
from pymlg.numpy import SO2, SE2, SE3, SO3
from scipy import stats
from typing import Hashable, Tuple
from navlie.types import State
from navlie.lib.states import SE2State
from navlie.batch.residuals import Residual
from mixtures.solver import ProblemExtended
from mixtures.gaussian_mixtures import (
    MaxMixtureResidual,
    MaxSumMixtureResidual,
    HessianSumMixtureResidual,
    SumMixtureResidual,
    HessianSumMixtureResidualStandardCompatibility,
)
from navlie.lib.states import SE3State
import scipy.linalg as scp_linalg


class SinglePointPsrResidual(Residual):
    # 2D version of the residual
    m_i: np.ndarray
    p_j: np.ndarray
    Sigma_i: np.ndarray
    Sigma_j: np.ndarray
    finite_difference: bool

    def __init__(
        self,
        keys: List[Hashable],
        m_i: np.ndarray,
        p_j: np.ndarray,
        Sigma_i: np.ndarray,
        Sigma_j: np.ndarray,
        finite_difference=False,
    ):
        super().__init__(keys)
        self.m_i = m_i
        self.p_j = p_j
        self.Sigma_i = Sigma_i
        self.Sigma_j = Sigma_j
        self.finite_difference = finite_difference

    def evaluate(
        self,
        states: List[State],
        compute_jacobians: List[bool] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        x: SE2State = states[0]
        L = self.sqrt_info_matrix(states)

        C_ts, r_t_st = x.group.to_components(x.value)
        r_j_ts_check = C_ts @ self.m_i.reshape(-1, 1) + r_t_st.reshape(-1, 1)
        error = self.p_j.squeeze() - r_j_ts_check.squeeze()
        error = L.T @ error
        if np.isnan(error).any():
            raise Exception("NaN in the error")
        if compute_jacobians:
            jac_list = [None]
            if compute_jacobians[0]:
                if self.finite_difference:
                    jac_list = self.jacobian_fd(states, step_size=1e-8)
                else:
                    jac = L.T @ self.jacobian_unnormalized(C_ts, r_t_st)
                    jac_list = [jac]
            return error, jac_list
        return error

    def jacobian_unnormalized(self, C_ts: np.ndarray, r_t_st: np.ndarray):
        jac_phi = -np.array([[0, -1], [1, 0]]) @ (C_ts @ self.m_i + r_t_st)
        jac_phi = jac_phi.reshape(-1, 1)
        jac_r = -np.eye(2)
        jac = np.hstack([jac_phi, jac_r])
        return jac

    def sqrt_info_matrix(self, states: List[State]) -> np.ndarray:
        x: SE2State = states[0]
        C_ts, _ = x.group.to_components(x.value)
        # Overwrite cov in the what comes next
        cov = C_ts @ self.Sigma_j @ C_ts.T + self.Sigma_i
        # scp_linalg.inv(cov, overwrite_a=True)
        # scp_linalg.cholesky(cov, overwrite_a=True)
        # return cov
        return np.linalg.cholesky(np.linalg.inv(cov))


class SinglePointPsrResidual3d(SinglePointPsrResidual):
    def jacobian_unnormalized(self, C_ts: np.ndarray, r_t_st: np.ndarray):
        jac_phi = SO3.cross(r_t_st.reshape(-1, 1) + C_ts @ self.m_i.reshape(-1, 1))
        jac_r = -np.eye(3)
        jac = np.hstack([jac_phi, jac_r])
        return jac


def problem_setup(
    monte_carlo_landmark_generation_bounds: List[float],
    C_st: np.ndarray,
    r_s_ts: np.ndarray,
    cluster_size: int,
    cluster_spread: float,
    fraction_of_landmarks_to_cluster: float,
    num_landmarks: int,
    ref_noise_stddev_bounds: List[float],
    meas_noise_stddev_bounds: List[float],
    dims: int = 2,
):
    # this is obsolete but kept cause the unit tests use it
    # todo kill this off
    ref_landmarks = np.random.uniform(
        monte_carlo_landmark_generation_bounds[0],
        monte_carlo_landmark_generation_bounds[1],
        size=(2, num_landmarks),
    )
    ref_landmarks = duplicate_landmarks(
        ref_landmarks,
        cluster_size,
        cluster_spread,
        fraction_of_landmarks_to_cluster,
    )

    ref_covs = generate_covariances(
        ref_noise_stddev_bounds[0],
        ref_noise_stddev_bounds[1],
        ref_landmarks.shape[1],
    )
    source_covs = generate_covariances(
        meas_noise_stddev_bounds[0],
        meas_noise_stddev_bounds[1],
        ref_landmarks.shape[1],
    )
    source_landmarks = C_st @ ref_landmarks + r_s_ts.reshape(-1, 1)

    return ref_landmarks, source_landmarks, ref_covs, source_covs


def corrupt_landmarks_with_noise(landmarks_true: np.ndarray, covs: List[np.ndarray]):
    landmarks = [
        landmarks_true[:, lv1]
        + stats.multivariate_normal.rvs(
            mean=np.zeros(landmarks_true.shape[0]), cov=covs[lv1]
        )
        for lv1 in range(landmarks_true.shape[1])
    ]
    landmarks = np.hstack([l.reshape(-1, 1) for l in landmarks])
    return landmarks


def generate_random_transformation(
    angle_range: List[float], r_range: List[float], debug=False, dims=2
):
    phi_bounds = [lv1 / 180 * np.pi for lv1 in angle_range]
    if dims == 2:
        phi = np.random.uniform(*phi_bounds)
        r_t_st = np.random.uniform(*r_range, (2, 1))
        if debug:
            phi = 10 / 180 * np.pi
            r_t_st = 0.5
        C_ts = SO2.Exp(phi)
        T_ts = SE2.from_components(C_ts, r_t_st)
        T_st = SE2.inverse(T_ts)
        C_st, r_s_ts = SE2.to_components(T_st)
    if dims == 3:
        dxi_phi = np.random.uniform(*phi_bounds, size=3)
        dxi_r = np.random.uniform(*r_range, (3))
        if debug:
            dxi_phi = np.ones(3) * 10 / 180 * np.pi
            dxi_r = np.ones(3) * 0.5
        xi = np.concatenate([dxi_phi, dxi_r])
        T_ts = SE3.Exp(xi)
        T_st = SE3.inverse(T_ts)
        C_st, r_s_ts = SE3.to_components(T_st)
        C_ts, r_t_st = SE3.to_components(T_ts)
    return C_st, r_s_ts, C_ts, r_t_st


def generate_covariances(lbound: float, ubound: float, num_covs: int, dims: int):
    # Generate covariances without off-diagonal elements
    # then rotate them to get those off-diagonal elements and add some spice
    # to the problem

    group: SO2 = None
    if dims == 2:
        n_angles = 1
        group = SO2
    if dims == 3:
        n_angles = 3
        group = SO3
    stddevs = np.random.uniform(lbound, ubound, size=(dims, num_covs))
    angles = np.random.uniform(-np.pi, np.pi, size=(n_angles, num_covs))
    angles = [angles[:, lv1] for lv1 in range(num_covs)]
    stddevs = [stddevs[:, lv1] for lv1 in range(stddevs.shape[1])]

    def C_ab(angle):
        return group.Exp(angle)

    covs = [
        C_ab(angle) @ np.diag(stddev**2) @ C_ab(angle).T
        for angle, stddev in zip(angles, stddevs)
    ]
    return covs


def duplicate_landmarks(
    ref_landmarks: np.ndarray,
    cluster_size: int,
    cluster_spread: float,
    fraction_of_landmarks_to_cluster: float,
):
    ref_landmarks = [
        ref_landmarks[:, lv1].reshape(-1, 1) for lv1 in range(ref_landmarks.shape[1])
    ]

    landmark_duplicates_noiseless: List[np.ndarray] = random.sample(
        ref_landmarks,
        int(np.ceil(fraction_of_landmarks_to_cluster * len(ref_landmarks))),
    ) * (cluster_size - 1)

    landmark_duplicates = [
        l + np.random.normal(scale=cluster_spread, size=l.shape)
        for l in landmark_duplicates_noiseless
    ]
    ref_landmarks = ref_landmarks + landmark_duplicates
    ref_landmarks = np.hstack(ref_landmarks)
    return ref_landmarks


from typing import Dict


def solve_psr_problem(
    ref_landmarks: np.ndarray,
    source_landmarks: np.ndarray,
    ref_covs: List[np.ndarray],
    source_covs: List[np.ndarray],
    initial_guess=None,
    residual_type="MSM",
    finite_difference=False,
    solver_params: Dict = None,
    verbose=False,
):
    key = "x"
    dims = ref_landmarks.shape[0]
    res_type: SinglePointPsrResidual = None
    if dims == 2:
        state = SE2State
        res_type = SinglePointPsrResidual
        if initial_guess is None:
            initial_guess = np.eye(3)
    if dims == 3:
        state = SE3State
        res_type = SinglePointPsrResidual3d
        if initial_guess is None:
            initial_guess = np.eye(4)
    x = state(initial_guess, 0.0, key, direction="left")
    problem = ProblemExtended(
        solver=solver_params["solver"],
        max_iters=solver_params["max_iters"],
        step_tol=solver_params["step_tol"],
        gradient_tol=solver_params["gradient_tol"],
        ftol=solver_params["ftol"],
        tau=solver_params["tau"],
        verbose=verbose,
    )
    n_landmarks = ref_landmarks.shape[1]
    problem.add_variable(key, x)
    for lv1 in range(source_landmarks.shape[1]):
        # Add a residual for every source landmark (lv1)
        # Where components are possible data associations are to each reference landmark (lv2)
        component_residuals = []
        for lv2 in range(ref_landmarks.shape[1]):
            component_residuals.append(
                res_type(
                    [key],
                    source_landmarks[:, lv1],
                    ref_landmarks[:, lv2],
                    source_covs[lv1],
                    ref_covs[lv2],
                    finite_difference=finite_difference,
                )
            )
        weights = [1 / n_landmarks] * n_landmarks
        residual_dict = {
            "MM": MaxMixtureResidual(component_residuals, weights),
            "SM": SumMixtureResidual(component_residuals, weights),
            "MSM": MaxSumMixtureResidual(component_residuals, weights, 10),
            "HSM": HessianSumMixtureResidual(
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
                normalization_constant=solver_params["tau"],
            ),
        }
        problem.add_residual(residual_dict[residual_type])

    result = problem.solve()
    # Note: We end up solving for C_ts (from source to ref)
    # though we generate the source landmarks using
    # C_st.
    return result, problem
