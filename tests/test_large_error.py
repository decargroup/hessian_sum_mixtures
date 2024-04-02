import pytest
import numpy as np
from navlie.lib.states import VectorState
from mixtures.gaussian_mixtures import (
    GaussianMixtureResidual,
    HessianSumMixtureResidual,
)
from mixtures.vanilla_mixture.mixture_utils import get_components, create_residuals
from typing import Dict
from typing import List, Callable
from navlie.types import State

"""
There's an edge case when errors become very high, where their sum evaluates to zero, 
which results in a division by zero. This issue was fixed by normalizing by the smallest 
error, and this test makes sure of that. 
"""


class mockArgs:
    def __init__(self, dims, weights, means, covariances):
        self.weights = weights
        self.means = means
        self.covariances = covariances
        self.dims = dims


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
        test_points = [VectorState(np.array([1000000, 100000]), 0.0, "x")]
        for test_point in test_points:
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
            e_sm, _ = sm_res.evaluate([test_point], [True])
            jac_sm_list_wrt_err, info_dict_sm = sm_res.mix_jacobians(
                error_value_list,
                jacobian_list_of_lists,
                sqrt_info_matrix_list,
                output_details=True,
            )

            jac_loss_sm = (jac_sm_list_wrt_err[0].T @ e_sm.reshape(-1, 1)).T

            e_msm, jac_msm_list_wrt_err = msm_res.evaluate([test_point], [True])
            jac_loss_msm = (jac_msm_list_wrt_err[0].T @ e_msm.reshape(-1, 1)).T

            hessian_hsm_list, _ = hsm_res.compute_hessians([test_point], [True], True)
            hessian_hsm = hessian_hsm_list[0]
            assert not np.isnan(np.min(hessian_hsm))
            assert not np.isnan(np.min(jac_loss_sm))
            assert not np.isnan(np.min(jac_loss_msm))


if __name__ == "__main__":
    test_matching_jacobians()
