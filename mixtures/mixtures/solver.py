import navlie
from navlie.batch.losses import LossFunction
import sys
import functools
import itertools
import numpy as np
from navlie import State, StateWithCovariance
from navlie import MeasurementModel, Measurement
from typing import List, Tuple, Hashable
from navlie.batch.residuals import Residual
from mixtures.gaussian_mixtures import GaussianMixtureResidual

from typing import Dict, Hashable, List, Tuple
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from navlie.batch.estimator import Problem
from navlie.types import State
from abc import ABC, abstractmethod
from navlie.lib.states import VectorState
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, Process, Queue
import time


@dataclass
class ErrorResult:
    error: np.ndarray
    jacobian: np.ndarray
    hessian: np.ndarray
    cost: float


class ProblemExtended(Problem):
    """
    Allows Hessian approximations that are more involved than H \trans H
    """

    def __init__(
        self,
        solver: str = "GN",
        max_iters: int = 100,
        step_tol: float = 1e-8,
        ftol: float = 1e-10,
        gradient_tol: float = 1e-10,
        tau: float = 1e-11,
        verbose: bool = True,
        save_histories: bool = True,
        n_threads: int = 1,
        debug=False,
        processing_type="pool",
    ):
        super().__init__(
            solver=solver,
            max_iters=max_iters,
            step_tol=step_tol,
            tau=tau,
            verbose=verbose,
            save_histories=save_histories,
        )
        self.ftol = ftol
        self.gradient_tol = gradient_tol
        self.n_threads = n_threads
        self.debug = debug
        self.processing_type = processing_type

    def is_converged(self, delta_cost, cost, dx, grad_norm) -> bool:
        converged = False
        if delta_cost is not None and grad_norm is not None:
            if delta_cost is not None:
                rel_cost_change = 0.0
                if cost != 0:
                    rel_cost_change = delta_cost / cost

            if self.step_tol is not None and dx < self.step_tol:
                converged = True
            if self.ftol is not None and delta_cost is not None:
                if rel_cost_change < self.ftol:
                    converged = True
            if cost == 0.0:
                converged = True
            if dx == 0.0:
                converged = True
            if self.gradient_tol is not None and grad_norm is not None:
                if grad_norm < self.gradient_tol:
                    converged = True
        return converged

    def _solve_gauss_newton(self) -> Dict[Hashable, State]:
        """Solves the optimization problem using Gauss-Newton.

        Returns
        -------
        Dict[Hashable, State]
            New dictionary of optimized variables.
        """

        dx = 10
        delta_cost = None
        grad_norm = None

        iter_idx = 0
        cost_list = []

        e, H, Hessian_all, cost = self.compute_error_jac_cost()
        cost_list.append(cost)

        # Print initial cost
        if self.verbose:
            header = "Initial cost: " + str(cost)
            print(header)

        while iter_idx < self.max_iters:
            if self.is_converged(delta_cost, cost_list[-1], dx, grad_norm):
                break

            H_spr = sparse.csr_matrix(H)
            Hessian_spr = sparse.csr_matrix(Hessian_all)
            A = Hessian_spr
            b = H_spr.T @ e

            delta_x = sparse.linalg.spsolve(A, -b).reshape((-1, 1))
            if np.isnan(delta_x).any():
                raise Exception("NaN in the delta_x")
            # Update the variables
            self._correct_states(delta_x)

            e, H, Hessian_all, cost = self.compute_error_jac_cost()
            cost_list.append(cost)

            dx = np.linalg.norm(delta_x)
            delta_cost = np.abs(cost_list[-1] - cost_list[-2])
            grad_norm = np.max(np.abs((e.T @ H).squeeze()))

            # if self.verbose:
            # self._display_header(iter_idx, cost, dx)
            if self.verbose:
                rel_cost_change = 0.0
                if cost_list[-1] != 0:
                    rel_cost_change = delta_cost / cost_list[-1]
                self._display_header(
                    iter_idx,
                    cost,
                    dx,
                    delta_cost,
                    rel_cost_change,
                    grad_norm,
                )

            iter_idx += 1

        # After convergence, compute final value of the cost function
        e, H, Hessian_all, cost = self.compute_error_jac_cost()
        cost_list.append(cost)

        self._cost_history = np.array(cost_list).reshape((-1))
        self._entire_cost_history = np.array(cost_list).reshape((-1))
        self._error_jacobian = H
        self._information_matrix = Hessian_all

        return self.variables

    def _solve_LM(self) -> Dict[Hashable, State]:
        """Solves the optimization problem using Gauss-Newton.

        Returns
        -------
        Dict[Hashable, State]
            New dictionary of optimized variables.
        """

        e, H, Hessian_all, cost = self.compute_error_jac_cost()
        delta_cost = None
        rel_cost_decrease = None
        grad_norm = None
        cost_list = [cost]
        entire_cost_list = [cost]

        H_spr = sparse.csr_matrix(H)
        Hessian_spr = sparse.csr_matrix(Hessian_all)
        A = Hessian_spr
        b = H_spr.T @ e

        iter_idx = 0
        dx = 10
        mu = self.tau * np.amax(A.diagonal())
        nu = 2
        prev_cost = cost

        if self.verbose:
            header = "Initial cost: " + str(cost)
            print(header)

        # Main LM loop

        while iter_idx < self.max_iters:

            A_solve = A + mu * sparse.identity(A.shape[0])
            delta_x = sparse.linalg.spsolve(A_solve, -b).reshape((-1, 1))
            dx = np.linalg.norm(delta_x)

            if self.is_converged(delta_cost, cost_list[-1], dx, grad_norm):
                break
            if np.isnan(delta_x).any():
                raise Exception("NaN in the delta_x")
            variables_test = {k: v.copy() for k, v in self.variables.items()}

            # Update the variables
            self._correct_states(delta_x, variables_test)

            # Compute the new value of the cost function after the update
            e, H, Hessian_all, cost = self.compute_error_jac_cost(
                variables=variables_test
            )

            gain_ratio = (prev_cost - cost) / (0.5 * delta_x.T @ (mu * delta_x - b))
            gain_ratio = gain_ratio.item(0)

            # If the gain ratio is above zero, accept the step
            if gain_ratio > 0:
                self.variables = variables_test
                mu = mu * max(1.0 / 3.0, 1.0 - (2.0 * gain_ratio - 1) ** 3)
                nu = 2

                e, H, Hessian_all, cost = self.compute_error_jac_cost()
                cost_list.append(cost)
                entire_cost_list.append(cost)
                prev_cost = cost

                H_spr = sparse.csr_matrix(H)
                Hessian_spr = sparse.csr_matrix(Hessian_all)
                A = Hessian_spr
                b = H_spr.T @ e
                status = "Accepted."
            else:
                entire_cost_list.append(cost)
                mu = mu * nu
                nu = 2 * nu
                status = "Rejected."

            if len(cost_list) >= 2:
                delta_cost = np.abs(cost_list[-1] - cost_list[-2])
                if cost_list[-1] != 0:
                    rel_cost_decrease = delta_cost / cost_list[-1]
                else:
                    rel_cost_decrease = 0
            grad_norm = np.max(np.abs((e.T @ H).squeeze()))

            if self.verbose:
                self._display_header(
                    iter_idx,
                    cost,
                    dx,
                    delta_cost,
                    rel_cost_decrease,
                    grad_norm,
                    status=status,
                )

            iter_idx += 1

        # After convergence, compute final value of the cost function
        e, H, Hessian_all, cost = self.compute_error_jac_cost()
        cost_list.append(cost)

        self._cost_history = np.array(cost_list).reshape((-1))
        self._entire_cost_history = np.array(entire_cost_list).reshape((-1))
        self._information_matrix = Hessian_all
        self._error_jacobian = H
        return self.variables

    def _display_header(
        self,
        iter_idx: int,
        current_cost: float,
        dx: float,
        delta_cost: float = None,
        delta_cost_rel: float = None,
        grad_norm: float = None,
        status: str = None,
    ):
        """Displays the optimization progress.

        Parameters
        ----------
        iter_idx : int
            Iteration number.
        current_cost : float
            Current objective function cost.
        dx : float
            Norm of step size.
        status : str, optional
            Status for LM, by default None
        """
        header = ("Iter: {0} || Cost: {1:.4e} || Step size: {2:.4e}").format(
            iter_idx, current_cost, dx
        )
        if delta_cost is not None:
            header += " || dC: {0:.4e}".format(delta_cost)
        if delta_cost_rel is not None:
            header += " || dC/C: {0:.4e}".format(delta_cost_rel)
        if grad_norm is not None:
            header += " || |grad|_inf: {0:.4e}".format(grad_norm)
        if status is not None:
            header += " || Status: " + status

        print(header)

    def compute_error_jac_cost(
        self, variables: Dict[Hashable, State] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Extends base method to compute Hessians too.

        Parameters
        ----------
        variables : Dict[Hashable, State], optional
            Variables, by default None. If None, uses the variables stored in
            the optimizer.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, float]
            Error vector, Jacobian, Hessian, and cost.
        """

        if variables is None:
            variables = self.variables

        # Compute the size of the problem if needed
        if self._size_errors is None or self._size_state is None:
            self._compute_size_of_problem()

        error_results: List[ErrorResult] = []
        for i, (residual, loss) in enumerate(zip(self.residual_list, self.loss_list)):
            error_results.append(
                compute_error_result(
                    residual, loss, variables, self.constant_variable_keys
                )
            )

        e, H, Hessian_all, cost = assemble_matrices(
            error_results,
            self.residual_list,
            self.residual_slices,
            self.variable_slices,
            self._size_errors,
            self._size_state,
        )
        return e, H, Hessian_all, cost


def compute_error_result(
    residual: Residual,
    loss: LossFunction,
    variables: Dict[str, State],
    constant_variable_keys: List[Hashable],
):
    variables_list = [variables[key] for key in residual.keys]
    residual: GaussianMixtureResidual = residual
    # Do not compute Jacobian for variables that are held fixed
    compute_jacobians = [
        False if key in constant_variable_keys else True for key in residual.keys
    ]

    # Evaluate current factor at states
    error, jacobians = residual.evaluate(variables_list, compute_jacobians)
    residual.jacobian_cache = jacobians
    hessians = residual.compute_hessians(
        variables_list,
        compute_hessians=compute_jacobians,
        use_jacobian_cache=True,
    )
    u = np.linalg.norm(error)
    cost = np.sum(loss.loss(u))

    return ErrorResult(error, jacobians, hessians, cost)


def assemble_matrices(
    error_results: List[ErrorResult],
    residual_list: List[Residual],
    residual_slices: List[slice],
    variable_slices: List[slice],
    size_errors: int,
    size_state: int,
):
    # Initialize the error vector, Jacobian and Hessian
    e = np.zeros((size_errors,))
    H = np.zeros((size_errors, size_state))
    Hessian_all = np.zeros((size_state, size_state))
    cost_list = []

    cost_list = [res.cost for res in error_results]
    cost = np.sum(np.array(cost_list))
    t0 = time.time()
    for i, (residual, error_result) in enumerate(zip(residual_list, error_results)):
        error = error_result.error
        jacobians = error_result.jacobian
        hessians = error_result.hessian  # Nested list.

        # sqrt_loss_weight = np.sqrt(loss.weight(u))
        sqrt_loss_weight = 1
        weighted_error = sqrt_loss_weight * error

        # Place errors
        e[residual_slices[i]] = weighted_error.ravel()

        for j, key in enumerate(residual.keys):
            jacobian = jacobians[j]
            if jacobian is not None:
                H[residual_slices[i], variable_slices[key]] = jacobian

        for lv1, key1 in enumerate(residual.keys):
            for lv2, key2 in enumerate(residual.keys):
                hessian = hessians[lv1][lv2]
                if hessian is None:
                    continue
                if not hessian.any():
                    continue
                Hessian_all[variable_slices[key1], variable_slices[key2]] = (
                    Hessian_all[variable_slices[key1], variable_slices[key2]] + hessian
                )
    # Sum up costs from each residual
    cost = np.sum(np.array(cost_list))

    return e.reshape((-1, 1)), H, Hessian_all, cost


def solve_vector_mixture_problem(
    gm_res: GaussianMixtureResidual,
    initial_guess: np.ndarray,
    state_key: str,
    solver="LM",
    verbose=False,
):
    key = state_key
    x = VectorState(initial_guess, 0.0, key)
    problem = ProblemExtended(solver=solver, verbose=verbose)
    problem.add_residual(gm_res)
    problem.add_variable(key, x)
    result = problem.solve()
    return result
