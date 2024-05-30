from typing import List

import numpy as np

from navlie import State
from navlie.batch.gaussian_mixtures import SumMixtureResidual
from navlie.batch.residuals import Residual, split_up_hessian_by_state
from typing import Tuple


class HessianSumMixtureResidualDirectHessian(SumMixtureResidual):
    """
    The HessianSumMixtureResidual that directly specifies the Hessian approximation
    for the Gaussian Mixture.
    """

    use_triggs: bool
    ceres_triggs_patch: bool

    def __init__(
        self,
        errors: List[Residual],
        weights,
        use_triggs: bool = False,
        ceres_triggs_patch: bool = False,
    ):
        super().__init__(errors, weights)
        self.use_triggs = use_triggs
        self.ceres_triggs_patch = ceres_triggs_patch

    def compute_loss_jacobian(
        self,
        error_value_list: List[np.ndarray],
        jacobian_list: List[
            np.ndarray
        ],  # List of Jacobians of each error component wrt state
        sqrt_info_matrix_list: List[np.ndarray],
        output_details=False,
    ) -> List[np.ndarray]:
        alpha_list = [
            weight * np.linalg.det(sqrt_info_matrix)
            for weight, sqrt_info_matrix in zip(self.weights, sqrt_info_matrix_list)
        ]

        # Remove extraneous dimension.
        error_value_list = [e.reshape(-1, 1) for e in error_value_list]

        eTe_list = [e.T @ e for e in error_value_list]
        eTe_dom = min(eTe_list)
        sum_exp = np.sum(
            [
                alpha * np.exp(-0.5 * e.T @ e + 0.5 * eTe_dom)
                for alpha, e in zip(alpha_list, error_value_list)
            ]
        )

        drho_df_list = [
            alpha * np.exp(0.5 * eTe_dom - 0.5 * eTe) / sum_exp
            for alpha, eTe in zip(alpha_list, eTe_list)
        ]

        f_i_jac_list = [e.T @ dedx for e, dedx in zip(error_value_list, jacobian_list)]

        n_x = f_i_jac_list[0].shape[1]
        numerator = np.zeros((1, n_x))

        numerator_list = [
            drho * f_i_jac for drho, f_i_jac in zip(drho_df_list, f_i_jac_list)
        ]

        for term in numerator_list:
            numerator += term

        numerator = numerator

        jac = numerator
        if not output_details:
            return jac

        else:
            info_dict = {
                "alphas": alpha_list,
                "sqrt_info_matrices": sqrt_info_matrix_list,
                "sum_exp": [sum_exp],
                "drho_df_list": drho_df_list,
                "f_i_jac_list": f_i_jac_list,
                "error_value_list": error_value_list,
                "jacobian_list": jacobian_list,
                "numerator_list": numerator_list,
                "numerator": numerator,
            }
            return jac, info_dict

    def compute_hessian_from_error_jac(
        self,
        error_value_list: List[np.ndarray],
        jacobian_list: List[np.ndarray],
        sqrt_info_matrix_list: List[np.ndarray],
        triggs_correction: bool = False,
        ceres_triggs_patch: bool = False,
        output_details: bool = False,
    ) -> List[np.ndarray]:

        alpha_list = [
            weight * np.linalg.det(sqrt_info_matrix)
            for weight, sqrt_info_matrix in zip(self.weights, sqrt_info_matrix_list)
        ]
        error_value_list = [e.reshape(-1, 1) for e in error_value_list]
        eTe_list = [e.T @ e for e in error_value_list]

        # Normalize all the exponent arguments to avoid numerical issues.
        eTe_dom = min(eTe_list)
        exp_list = [np.exp(0.5 * eTe_dom - 0.5 * e.T @ e) for e in error_value_list]
        sum_exp = np.sum(
            [
                alpha * np.exp(0.5 * eTe_dom - 0.5 * e.T @ e)
                for alpha, e in zip(alpha_list, error_value_list)
            ]
        )

        drho_df_list = [
            alpha * np.exp(0.5 * eTe_dom - 0.5 * eTe) / sum_exp
            for alpha, eTe in zip(alpha_list, eTe_list)
        ]

        f_i_jac_list = [e.T @ dedx for e, dedx in zip(error_value_list, jacobian_list)]
        #        f_i_hessian_list = [2 * H.T @ H for H in jacobian_list]
        f_i_hessian_list = [H.T @ H for H in jacobian_list]

        term1_components = [
            drho * f_i_hessian
            for drho, f_i_hessian in zip(drho_df_list, f_i_hessian_list)
        ]

        if triggs_correction:
            K = len(error_value_list)
            term2_components = [0.0] * len(term1_components)
            for k in range(K):
                term2_comp_k = np.zeros(term1_components[0].shape)
                for j in range(K):
                    # d2rho: \partial rho \partial f_j \partial f_k
                    d2rho = alpha_list[k] * alpha_list[j] * exp_list[k] * exp_list[j]
                    if j == k:
                        d2rho = d2rho - alpha_list[k] * exp_list[k] * sum_exp

                    if ceres_triggs_patch and d2rho < 0.0:
                        d2rho = 0.0
                    d2rho = d2rho / sum_exp**2
                    term2_comp_k = (
                        term2_comp_k + d2rho * f_i_jac_list[k].T @ f_i_jac_list[j]
                    )
                term2_components[k] = term2_comp_k
        else:
            term2_components = [0.0] * len(term1_components)

        component_hessians = [
            term1 + term2 for term1, term2 in zip(term1_components, term2_components)
        ]

        hessian = np.zeros(component_hessians[0].shape)
        for comp in component_hessians:
            hessian += comp
        hessian = hessian.T
        if np.isnan(hessian).any():
            raise Exception("NaN in the Hessian")
        if not output_details:
            return hessian
        else:
            info_dict = {
                "alphas": alpha_list,
                "sqrt_info_matrices": sqrt_info_matrix_list,
                "exp_list": exp_list,
                "sum_exp": sum_exp,
                "drho_df_list": drho_df_list,
                "f_i_jac_list": f_i_jac_list,
                "f_i_hessian_list": f_i_hessian_list,
                "error_value_list": error_value_list,
                "jacobian_list": jacobian_list,
                "term1_components": term1_components,
                "term2_components": term2_components,
            }
            return hessian, info_dict

    def compute_hessians(
        self,
        states: List[State],
        compute_hessians: List[bool],
        output_jacs: bool = False,
        use_jacobian_cache=False,
    ) -> List[np.ndarray]:
        if not use_jacobian_cache:
            (
                error_value_list,
                jacobian_list_of_lists,
                sqrt_info_matrix_list,
            ) = self.evaluate_component_residuals(
                states, compute_jacobians=[True] * len(states)
            )
        else:
            error_value_list = self.error_value_list_cache
            jacobian_list_of_lists = self.jacobian_list_of_lists_cache
            sqrt_info_matrix_list = self.sqrt_info_matrix_list_cache

        # Stack all the state jacobians for every residual.
        jacobian_full_list = [
            np.hstack([jac for jac in jac_list if jac is not None])
            for jac_list in jacobian_list_of_lists
        ]

        # jacs_res: For given residual. Contains jacobians w.r.t. all states
        hessian = self.compute_hessian_from_error_jac(
            error_value_list,
            jacobian_full_list,
            sqrt_info_matrix_list,
            self.use_triggs,
            self.ceres_triggs_patch,
        )
        hessians = split_up_hessian_by_state(states, hessian, compute_hessians)
        if output_jacs:
            jac_loss_list = [None for lv1 in range(len(states))]
            for lv1 in range(len(states)):
                jacobian_list = [jac_list[lv1] for jac_list in jacobian_list_of_lists]
                jac_loss_list[lv1] = self.compute_loss_jacobian(
                    error_value_list,
                    jacobian_list,
                    sqrt_info_matrix_list,
                )
        if not output_jacs:
            return hessians
        if output_jacs:
            return hessians, jac_loss_list
