from typing import Hashable, List, Tuple
import numpy as np
from navlie import State
from navlie.batch.residuals import Residual, split_up_hessian_by_state


class GaussianMixtureResidual(Residual):
    # Errors that the mixture holds on to, each with its own state keys
    errors: List[Residual]
    # Keys of states that get passed into evaluate method
    keys: List[Hashable]
    # Weights of the Gaussians
    weights: List[float]
    # jacobian_cache
    # error wrt state
    jacobian_cache: List[np.ndarray] = None

    # Caching Mixture-specific values for efficiency
    error_value_list_cache: List[np.ndarray] = None
    # jacobian_list_of_lists_cache
    # jacobians with respect to each state for each component
    jacobian_list_of_lists_cache: List[List[np.ndarray]] = None
    # sqrt_info_matrix_list_cache
    sqrt_info_matrix_list: List[np.ndarray] = None
    """
    For Max-Mixture, Sum-Mixture, and Max-Sum-Mixture the following reference is used extensively: 
    @ARTICLE{9381625,
        author={Pfeifer, Tim and Lange, Sven and Protzel, Peter},
        journal={IEEE Robotics and Automation Letters}, 
        title={Advancing Mixture Models for Least Squares Optimization}, 
        year={2021},
        volume={6},
        number={2},
        pages={3941-3948},
        doi={10.1109/LRA.2021.3067307}}
    """

    def __init__(self, errors: List[Residual], weights):
        self.errors = errors
        self.keys = []
        for error in errors:
            for key in error.keys:
                if key not in self.keys:
                    self.keys.append(key)
        self.weights = weights / np.sum(np.array(weights))

    def mix_errors(
        self,
        error_value_list: List[np.ndarray],
        sqrt_info_matrix_list: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Each mixture must implement this method..
        Compute the factor error from the errors corresponding to each component

        All errors are assumed to be normalized and have identity covariance.
        Parameters
        ----------
        error_value_list : List[np.ndarray],
            List of errors corresponding to each component
        """
        pass

    def mix_jacobians(
        self,
        error_value_list: List[np.ndarray],
        jacobian_list_of_lists: List[List[np.ndarray]],
        sqrt_info_matrix_list: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Each mixture must implement this method.
        For every state, compute Jacobian of the Gaussian mixture w.r.t. that state

        Parameters
        ----------
        error_value_list : List[np.ndarray],
            List of errors corresponding to each component
        jacobian_list : List[List[np.ndarray]]
            Outer list corresponds to each component, for each of which the inner list contains
            the component Jacobians w.r.t. every state.
        """
        pass

    def evaluate_component_residuals(
        self,
        states: List[State],
        compute_jacobians: List[bool] = None,
    ) -> Tuple[List[np.ndarray], List[List[np.ndarray]], List[np.ndarray]]:
        error_value_list: List[np.ndarray] = []
        jacobian_list_of_lists: List[List[np.ndarray]] = []
        sqrt_info_matrix_list: List[np.ndarray] = []

        for error in self.errors:
            cur_keys = error.keys
            key_indices = [self.keys.index(cur_key) for cur_key in cur_keys]
            cur_states = [states[key_idx] for key_idx in key_indices]

            if compute_jacobians:
                # The error and jacobians returned by the sub-error.
                # These need to be put together keeping in mind that
                # the overall mixture error depends on the union of all the states.
                cur_compute_jacobians = [
                    compute_jacobians[key_idx] for key_idx in key_indices
                ]
                val, jac_list_subset = error.evaluate(cur_states, cur_compute_jacobians)
                n_e = val.shape[0]  # Error dimension.

                # Jacobians of states that are not to be computed are set to zero.
                # Jacobians of states that are to be computed, but
                # the state of which is not one the error depends on,
                # are set to zero.
                jac_list_all_states = [None for lv1 in range(len(states))]

                # Set relevant Jacobians to zero first. Then
                # overwrite those that the error depends on.
                for lv1, (compute_jac, state) in enumerate(
                    zip(compute_jacobians, states)
                ):
                    if compute_jac:
                        jac_list_all_states[lv1] = np.zeros((n_e, state.dof))

                # jac_list_subset only has elements corresponding to the states that the error
                # is dependent on.
                # We need to put them in the right place in the list of jacobians
                # that correspond to the whole state list..
                for key_idx, jac in zip(key_indices, jac_list_subset):
                    jac_list_all_states[key_idx] = jac

                jacobian_list_of_lists.append(jac_list_all_states)
            else:
                val = error.evaluate(cur_states)

            error_value_list.append(val)
            sqrt_info_matrix_list.append(error.sqrt_info_matrix(cur_states))
        self.error_value_list_cache = error_value_list
        self.jacobian_list_of_lists_cache = jacobian_list_of_lists
        self.sqrt_info_matrix_list = sqrt_info_matrix_list

        return error_value_list, jacobian_list_of_lists, sqrt_info_matrix_list

    def evaluate(
        self,
        states: List[State],
        compute_jacobians: List[bool] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        (
            error_value_list,
            jacobian_list_of_lists,
            sqrt_info_matrix_list,
        ) = self.evaluate_component_residuals(states, compute_jacobians)
        e = self.mix_errors(error_value_list, sqrt_info_matrix_list)
        if compute_jacobians:
            jac_list = self.mix_jacobians(
                error_value_list, jacobian_list_of_lists, sqrt_info_matrix_list
            )
            self.jacobian_cache = jac_list
            return e, jac_list
        # print("Error value")
        # print(e)
        return e


class MaxMixtureResidual(GaussianMixtureResidual):
    def mix_errors(
        self,
        error_value_list: List[np.ndarray],
        sqrt_info_matrix_list: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        alphas = [
            weight * np.linalg.det(sqrt_info_matrix)
            for weight, sqrt_info_matrix in zip(self.weights, sqrt_info_matrix_list)
        ]
        # Maximum component obtained as
        # K = argmax alpha_k exp(-0.5 e^\trans e)
        #   = argmin -2* log alpha_k + e^\trans e
        res_values = np.array(
            [
                -np.log(alpha) + 0.5 * e.T @ e
                for alpha, e in zip(alphas, error_value_list)
            ]
        )
        dominant_idx = np.argmin(res_values)
        linear_part = error_value_list[dominant_idx]

        alpha_k = alphas[dominant_idx]
        alpha_max = max(alphas)

        nonlinear_part = np.array(np.log(alpha_max / alpha_k)).reshape(-1)
        nonlinear_part = np.sqrt(2) * np.sqrt(nonlinear_part)
        e_mix = np.concatenate([linear_part, nonlinear_part])
        return e_mix

    def mix_jacobians(
        self,
        error_value_list: List[np.ndarray],
        jacobian_list_of_lists: List[List[np.ndarray]],
        sqrt_info_matrix_list: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        alphas = [
            weight * np.linalg.det(sqrt_info_matrix)
            for weight, sqrt_info_matrix in zip(self.weights, sqrt_info_matrix_list)
        ]

        res_values = np.array(
            [
                -np.log(alpha) + 0.5 * e.T @ e
                for alpha, e in zip(alphas, error_value_list)
            ]
        )
        dominant_idx = np.argmin(res_values)
        jac_list_linear_part: List[np.ndarray] = jacobian_list_of_lists[dominant_idx]

        jac_list = []
        for jac in jac_list_linear_part:
            if jac is not None:
                jac_list.append(np.vstack([jac, np.zeros((1, jac.shape[1]))]))
            else:
                jac_list.append(None)
        return jac_list


class MaxSumMixtureResidual(GaussianMixtureResidual):
    """
    Pfeifer's approach.
    """

    damping_const: float

    def __init__(self, errors: List[Residual], weights, damping_const: float = 10):
        super().__init__(errors, weights)
        self.damping_const = damping_const

    def mix_errors(
        self,
        error_value_list: List[np.ndarray],
        sqrt_info_matrix_list: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        alphas = [
            weight * np.linalg.det(sqrt_info_matrix)
            for weight, sqrt_info_matrix in zip(self.weights, sqrt_info_matrix_list)
        ]

        # Linear part is the same as for the max-mixture
        # Maximum component obtained as
        # K = argmax alpha_k exp(-0.5 e^\trans e)
        #   = argmin -2* log alpha_k + e^\trans e
        res_values = np.array(
            [
                -np.log(alpha) + 0.5 * e.T @ e
                for alpha, e in zip(alphas, error_value_list)
            ]
        )
        dominant_idx = np.argmin(res_values)
        linear_part = error_value_list[dominant_idx]

        alpha_max = max(alphas)

        # Nonlinear part changes quite a bit. Very similar to sum-mixture.
        err_kmax = error_value_list[dominant_idx]
        scalar_errors_differences = [
            -0.5 * e.T @ e + 0.5 * err_kmax.T @ err_kmax for e in error_value_list
        ]

        nonlinear_part = self.compute_nonlinear_part(scalar_errors_differences, alphas)
        e_mix = np.concatenate([linear_part, nonlinear_part])

        return e_mix

    def compute_nonlinear_part(
        self, scalar_errors_differences: List[np.ndarray], alphas: List[float]
    ):
        alpha_max = max(alphas)
        normalization_const = np.log(len(alphas) * alpha_max + self.damping_const)

        sum_term = np.log(
            np.sum(
                np.array(
                    [
                        alpha * np.exp(e)
                        for alpha, e in zip(alphas, scalar_errors_differences)
                    ]
                )
            )
        )
        nonlinear_part = np.sqrt(2) * np.sqrt(normalization_const - sum_term)
        nonlinear_part = np.array(nonlinear_part).reshape(-1)
        return nonlinear_part

    def mix_jacobians(
        self,
        error_value_list: List[np.ndarray],
        jacobian_list_of_lists: List[List[np.ndarray]],
        sqrt_info_matrix_list: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        n_state_list = len(jacobian_list_of_lists[0])

        alphas = [
            weight * np.linalg.det(sqrt_info_matrix)
            for weight, sqrt_info_matrix in zip(self.weights, sqrt_info_matrix_list)
        ]

        # LINEAR PART
        res_values = np.array(
            [
                -np.log(alpha) + 0.5 * e.T @ e
                for alpha, e in zip(alphas, error_value_list)
            ]
        )
        dominant_idx = np.argmin(res_values)
        jac_list_linear_part: List[np.ndarray] = jacobian_list_of_lists[dominant_idx]

        err_kmax = error_value_list[dominant_idx]

        scalar_errors_differences = [
            -0.5 * e.T @ e + 0.5 * err_kmax.T @ err_kmax for e in error_value_list
        ]

        # NONLINEAR PART
        # Compute error
        e_nl = self.compute_nonlinear_part(scalar_errors_differences, alphas)

        # Loop through every state to compute Jacobian with respect to it.
        jac_list_nl = []
        for lv1 in range(n_state_list):
            jacobian_list_components_wrt_cur_state = [
                jac_list[lv1] for jac_list in jacobian_list_of_lists
            ]
            if jacobian_list_components_wrt_cur_state[0] is not None:
                jac_dom = jacobian_list_components_wrt_cur_state[dominant_idx]
                n_x = jacobian_list_components_wrt_cur_state[0].shape[1]
                numerator = np.zeros((1, n_x))
                denominator = 0.0
                numerator_list = [
                    -alpha
                    * np.exp(scal_err)
                    * (
                        e_k.reshape(1, -1) @ -jac_e_i
                        + err_kmax.reshape(1, -1) @ jac_dom
                    )
                    for alpha, scal_err, e_k, jac_e_i in zip(
                        alphas,
                        scalar_errors_differences,
                        error_value_list,
                        jacobian_list_components_wrt_cur_state,
                    )
                ]
                denominator_list = [
                    alpha * np.exp(scal_err)
                    for alpha, scal_err in zip(alphas, scalar_errors_differences)
                ]

                for term in numerator_list:
                    numerator += term

                for term in denominator_list:
                    denominator += term
                denominator = denominator * e_nl
                jac_list_nl.append(numerator / denominator)
            else:
                jac_list_nl.append(None)

        jac_list = []
        for jac_lin, jac_nl in zip(jac_list_linear_part, jac_list_nl):
            if jac_nl is not None:
                jac_list.append(np.vstack([jac_lin, jac_nl]))
            else:
                jac_list.append(None)

        return jac_list


class SumMixtureResidual(GaussianMixtureResidual):
    def mix_errors(
        self,
        error_value_list: List[np.ndarray],
        sqrt_info_matrix_list: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        alphas = [
            weight * np.linalg.det(sqrt_info_matrix)
            for weight, sqrt_info_matrix in zip(self.weights, sqrt_info_matrix_list)
        ]
        normalization_const = sum(alphas)

        # K = argmax alpha_k exp(-0.5 e^\trans e)
        #   = argmin -2* log alpha_k + e^\trans e

        res_values = np.array(
            [
                -np.log(alpha) + 0.5 * np.linalg.norm(e) ** 2
                for alpha, e in zip(alphas, error_value_list)
            ]
        )
        kmax = np.argmin(res_values)

        scalar_errors = np.array(
            [0.5 * np.linalg.norm(e) ** 2 for alpha, e in zip(alphas, error_value_list)]
        )

        sum_term = np.log(
            np.sum(
                np.array(
                    [
                        alpha * np.exp(-e + scalar_errors[kmax])
                        for alpha, e in zip(alphas, scalar_errors)
                    ]
                )
            )
        )
        e = np.sqrt(2) * np.sqrt(normalization_const + scalar_errors[kmax] - sum_term)
        return e

    def mix_jacobians(
        self,
        error_value_list: List[np.ndarray],
        jacobian_list_of_lists: List[
            List[np.ndarray]
        ],  # outer list is components, inner list states
        sqrt_info_matrix_list: List[np.ndarray],
        output_details=False,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        n_state_list = len(jacobian_list_of_lists[0])
        alpha_list = [
            weight * np.linalg.det(sqrt_info_matrix)
            for weight, sqrt_info_matrix in zip(self.weights, sqrt_info_matrix_list)
        ]
        e_sm = self.mix_errors(error_value_list, sqrt_info_matrix_list)
        error_value_list = [e.reshape(-1, 1) for e in error_value_list]
        eTe_list = [e.T @ e for e in error_value_list]
        eTe_dom = min(eTe_list)
        exp_list = [np.exp(-0.5 * e.T @ e + 0.5 * eTe_dom) for e in error_value_list]
        sum_exp = np.sum(
            [
                alpha * np.exp(-0.5 * e.T @ e + 0.5 * eTe_dom)
                for alpha, e in zip(alpha_list, error_value_list)
            ]
        )

        drho_df_list = [
            alpha * exp / sum_exp for alpha, exp in zip(alpha_list, exp_list)
        ]

        # Loop through every state to compute Jacobian with respect to it.
        jac_list = []
        for lv1 in range(n_state_list):
            jacobian_list_components_wrt_cur_state = [
                jac_list[lv1] for jac_list in jacobian_list_of_lists
            ]
            if jacobian_list_components_wrt_cur_state[0] is not None:
                n_x = jacobian_list_components_wrt_cur_state[0].shape[1]

                f_i_jac_list = [
                    e.T @ dedx
                    for e, dedx in zip(
                        error_value_list, jacobian_list_components_wrt_cur_state
                    )
                ]

                numerator = np.zeros((1, n_x))

                numerator_list = [
                    drho * f_i_jac for drho, f_i_jac in zip(drho_df_list, f_i_jac_list)
                ]
                for term in numerator_list:
                    numerator += term
                numerator = numerator
                denominator = e_sm
                jac_list.append(numerator / denominator)
            else:
                jac_list.append(None)
        if not output_details:
            return jac_list
        if output_details:
            info_dict = {
                "alphas": alpha_list,
                "sqrt_info_matrices": sqrt_info_matrix_list,
                "exp_list": exp_list,
                "sum_exp": [sum_exp],
                "jacobian_list": jacobian_list_of_lists[0],
                "f_i_jac_list": f_i_jac_list,
                # "e_sm": [e_sm],
                "error_value_list": error_value_list,
                "numerator_list": numerator_list,
                "numerator": numerator,
            }
            return jac_list, info_dict


class HessianSumMixtureResidual(SumMixtureResidual):
    """
    Overwrite the Hessian computation method.
    The method exposed to the solver only requires states and
    compute_hessians, while itself calling
    the compute_hessians_from_errors_jacs method that corresponds
    to the equations in the doc.
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
        (
            error_value_list,
            jacobian_list_of_lists,
            sqrt_info_matrix_list,
        ) = self.evaluate_component_residuals(
            states, compute_jacobians=[True] * len(states)
        )
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


class HessianSumMixtureResidualStandardCompatibility(GaussianMixtureResidual):
    """
    The hessian sum-mixture method patched for compatibility with nonlinear least squares solvers.
    The idea is as follows:
        The Hessian, Jacobian, and Losses are all defined already. We want this to work with Gauss Newton,
        so to find H and e such that
            Hessian = H.T H
            Jacobian = H.T e
            Loss = e.T e.
        For the hessian sum mixture H and e satisfying the first two equations are pretty easy to find by inspection.
        But then the loss from e.T e is not the same as the loss corresponding to the original log likelihood.
        So the error is the split into
        [e_hess, e_loss] such that e_hess is responsible for the the Hessian/Jacobian
        and e_loss cancels outs its effect on the loss.
        The corresponding Jacobian is set to zero.

        Setting no_use_complex_numbers to True corresponds to the method proposed
        to maintain nonlinear least-squares solver compatiility in our paper.
    """

    sum_mixture_residual: SumMixtureResidual
    no_use_complex_numbers: bool
    normalization_constant: float

    def __init__(
        self,
        errors: List[Residual],
        weights,
        no_use_complex_numbers=True,
        normalization_constant=0.1,
    ):
        super().__init__(errors, weights)
        self.sum_mixture_residual = SumMixtureResidual(errors, weights)
        self.no_use_complex_numbers = no_use_complex_numbers
        self.normalization_constant = normalization_constant

    def mix_errors(
        self,
        error_value_list: List[np.ndarray],
        sqrt_info_matrix_list: List[np.ndarray],
    ) -> List[np.ndarray]:
        alpha_list = [
            weight * np.linalg.det(sqrt_info_matrix)
            for weight, sqrt_info_matrix in zip(self.weights, sqrt_info_matrix_list)
        ]
        error_value_list = [e.reshape(-1, 1) for e in error_value_list]
        eTe_list = [e.T @ e for e in error_value_list]

        # Normalize all the exponent arguments to avoid numerical issues.
        eTe_dom = min(eTe_list)

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

        hsm_error = np.vstack(
            [np.sqrt(drho) * e for drho, e in zip(drho_df_list, error_value_list)]
        ).squeeze()

        desired_loss = np.sum(
            self.sum_mixture_residual.mix_errors(
                error_value_list, sqrt_info_matrix_list
            )
            ** 2
        )

        if not self.no_use_complex_numbers:
            current_loss = np.sum(hsm_error**2)
            diff = np.array(np.emath.sqrt(desired_loss - current_loss))
            hsm_error = np.concatenate(
                [
                    hsm_error,
                    np.atleast_1d(np.array(diff)),
                ]
            )
        if self.no_use_complex_numbers:
            current_loss = np.sum(hsm_error**2)

            delta = self.normalization_constant + desired_loss - current_loss
            if delta < 0:
                self.normalization_constant = delta + 1
                delta = self.normalization_constant + desired_loss - current_loss

            diff = np.array(np.sqrt(delta))
            hsm_error = np.concatenate(
                [
                    hsm_error,
                    np.atleast_1d(np.array(diff)),
                ]
            )
        return hsm_error

    def mix_jacobians(
        self,
        error_value_list: List[np.ndarray],
        jacobian_list_of_lists: List[List[np.ndarray]],
        sqrt_info_matrix_list: List[np.ndarray],
    ) -> List[np.ndarray]:
        n_state_list = len(jacobian_list_of_lists[0])

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

        jac_list = []
        for lv1 in range(n_state_list):
            jacobian_list_components_wrt_cur_state = [
                jac_list[lv1] for jac_list in jacobian_list_of_lists
            ]
            if jacobian_list_components_wrt_cur_state[0] is not None:
                nx = jacobian_list_components_wrt_cur_state[0].shape[1]

                jac = np.vstack(
                    [
                        np.sqrt(drho) * jac
                        for drho, jac in zip(
                            drho_df_list, jacobian_list_components_wrt_cur_state
                        )
                    ]
                )
                jac = np.vstack([jac, np.zeros((1, nx))])

                jac_list.append(jac)
            else:
                jac_list.append(None)
        return jac_list
