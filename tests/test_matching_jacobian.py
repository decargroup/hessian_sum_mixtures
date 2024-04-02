import pytest
import numpy as np
from navlie.lib.states import VectorState
from mixtures.gaussian_mixtures import (
    GaussianMixtureResidual,
    HessianSumMixtureResidual,
)
from mixtures.vanilla_mixture.mixture_utils import get_components, create_residuals
import argparse
from typing import Dict
from typing import List, Callable
from navlie.types import State


class mockArgs:
    def __init__(self, dims, weights, means, covariances):
        self.weights = weights
        self.means = means
        self.covariances = covariances
        self.dims = dims


def preprocess_args(args):
    args.weights = [float(w) for w in args.weights]
    args.covariances = [float(w) for w in args.covariances]
    args.means = [float(w) for w in args.means]
    args.dims = int(args.dims)
    return args


def jacobian_fd(
    states: List[State], func: Callable, ny: int, step_size=1e-8
) -> List[np.ndarray]:
    """
    Calculates the model jacobian with finite difference.

    Parameters
    ----------
    states : List[State]
        Evaluation point of Jacobians, a list of states that
        the residual is a function of.
    ny: int
        dimension of output
    func: Callable[List[State]]
    Returns
    -------
    List[np.ndarray]
        A list of Jacobians of the measurement model with respect to each of the input states.
        For example, the first element of the return list is the Jacobian of the residual
        w.r.t states[0], the second element is the Jacobian of the residual w.r.t states[1], etc.
    """
    jac_list: List[np.ndarray] = [None] * len(states)

    # Compute the Jacobian for each state via finite difference
    for state_num, X_bar in enumerate(states):
        e_bar = func(states)
        size_error = ny
        jac_fd = np.zeros((size_error, X_bar.dof))

        for i in range(X_bar.dof):
            dx = np.zeros((X_bar.dof, 1))
            dx[i, 0] = step_size
            X_temp = X_bar.plus(dx)
            state_list_pert: List[State] = []
            for state in states:
                state_list_pert.append(state.copy())

            state_list_pert[state_num] = X_temp
            e_temp = func(state_list_pert)
            jac_fd[:, i] = (e_temp - e_bar).flatten() / step_size

        jac_list[state_num] = jac_fd

    return jac_list


def test_matching_jacobians():
    args1 = mockArgs(1, [1], [1], [3])
    args2 = mockArgs(1, [0.5, 0.5], [0, 0], [1, 1])
    args3 = mockArgs(2, [0.5, 0.5], [0.5, 1.5, 1.5, 0.5], [2, 0, 0, 2, 1, 0, 0, 1])
    args4 = mockArgs(
        2,
        [0.3, 0.3, 0.3],
        [1, 2, 3, 4, 5, 6],
        [0.5, 0, 0, 0.5, 0.2, 0, 0, 0.2, 1, 0, 0, 1],
    )

    # for args in [args1, args2, args3, args4]:
    for args in [args3]:
        dims = args.dims
        means, covariances = get_components(args.dims, args.means, args.covariances)
        n_x = means[0].shape[0]

        res_dict = create_residuals(
            args.weights,
            means,
            covariances,
        )
        sm_res: GaussianMixtureResidual = res_dict["SM"]
        msm_res: GaussianMixtureResidual = res_dict["MSM"]
        hsm_res: HessianSumMixtureResidual = res_dict["HSM"]
        # Use all the terms to compute Hessian. Performance is not good for this method,
        # but useful as a test since it gives the exact Hessian for linear errors.
        hsm_res.use_triggs = True
        hsm_std_res: GaussianMixtureResidual = res_dict["HSM_STD"]
        hsm_no_complex: GaussianMixtureResidual = res_dict["HSM_NO_COMPLEX"]

        test_points_arr = np.random.uniform(low=-20, high=20, size=(dims, 100))
        test_points = [
            VectorState(test_points_arr[:, lv1], 0.0, "x")
            for lv1 in range(test_points_arr.shape[1])
        ]
        for test_point in test_points:
            # test_point = VectorState(np.array([0.15170507, -19.94530591]), 0.0, "x")
            (
                error_value_list,
                jacobian_list_of_lists,
                sqrt_info_matrix_list,
            ) = sm_res.evaluate_component_residuals([test_point], [True])

            jac_loss_hsm, info_dict_hsm = hsm_res.compute_loss_jacobian(
                error_value_list,
                [jac_list[0] for jac_list in jacobian_list_of_lists],
                sqrt_info_matrix_list,
                True,
            )
            hessian_list, jac_list = hsm_res.compute_hessians(
                [test_point], [True], True
            )
            jac_loss_hsm_from_hessian = jac_list[0]
            e_sm, _ = sm_res.evaluate([test_point], [True])
            jac_sm_list_wrt_err = sm_res.mix_jacobians(
                error_value_list,
                jacobian_list_of_lists,
                sqrt_info_matrix_list,
            )

            jac_loss_sm = (jac_sm_list_wrt_err[0].T @ e_sm.reshape(-1, 1)).T

            e_msm, jac_msm_list_wrt_err = msm_res.evaluate([test_point], [True])
            jac_loss_msm = (jac_msm_list_wrt_err[0].T @ e_msm.reshape(-1, 1)).T

            e_hsm_std, jac_hsm_std_list_wrt_err = hsm_std_res.evaluate(
                [test_point], [True]
            )
            jac_loss_hsm_std = (
                jac_hsm_std_list_wrt_err[0].T @ e_hsm_std.reshape(-1, 1)
            ).T

            e_hsm_no_complex, jac_hsm_no_complex_list_wrt_err = hsm_std_res.evaluate(
                [test_point], [True]
            )
            jac_loss_hsm_no_complex = (
                jac_hsm_no_complex_list_wrt_err[0].T @ e_hsm_no_complex.reshape(-1, 1)
            ).T

            def loss(states: List[State]):
                e_sm_, _ = sm_res.evaluate(states, [True] * len(states))
                return 0.5 * e_sm_**2

            jac_true_list = jacobian_fd([test_point], loss, 1, step_size=1e-8)
            jac_true = jac_true_list[0]

            # Finite difference the Jacobians w.r.t. the loss (H.T @ e)
            assert np.linalg.norm(jac_loss_sm - jac_true, "fro") < 1e-3

            assert np.linalg.norm(jac_loss_msm - jac_true, "fro") < 1e-4
            assert (
                np.linalg.norm(jac_loss_hsm_from_hessian - jac_loss_hsm, "fro") < 1e-5
            )
            assert np.linalg.norm(jac_loss_sm - jac_loss_hsm, "fro") < 1e-4
            assert np.linalg.norm(jac_loss_sm - jac_true, "fro") < 1e-3
            assert np.linalg.norm(jac_loss_hsm_std - jac_true, "fro") < 1e-3
            assert np.linalg.norm(jac_loss_hsm_no_complex - jac_true, "fro") < 1e-3

            # Finite difference test of the loss Hessian - is exact using the correction term
            # in the approach that computes Hessian explicitly.
            def jacobian_func(states: List[State]):
                _, jac_list = hsm_res.compute_hessians(states, [True], True)

                return jac_list[0].T

            hessian_fd = jacobian_fd([test_point], jacobian_func, n_x)[0]
            hessian_hsm_list, _ = hsm_res.compute_hessians([test_point], [True], True)
            hessian_hsm = hessian_hsm_list[0][0]

            assert np.linalg.norm(hessian_fd - hessian_hsm, "fro") < 1e-5


if __name__ == "__main__":
    test_matching_jacobians()
