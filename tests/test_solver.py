from typing import List
import numpy as np
from mixtures.solver import ProblemExtended
from navlie.batch.problem import OptimizationSummary, Problem
from mixtures.gaussian_mixtures import (
    GaussianMixtureResidual,
    MaxMixtureResidual,
    SumMixtureResidual,
    MaxSumMixtureResidual,
    HessianSumMixtureResidual,
    HessianSumMixtureResidualStandardCompatibility,
)
from mixtures.vanilla_mixture.mixture_utils import get_component_residuals
from mixtures.vanilla_mixture.monte_carlo import GaussianMixtureParameters
from navlie.lib.states import VectorState

"""
An extended solver is used for these problems, where the user is allowed
to specify a Hessian. This is ONLY done to compare 
the method where we specify Hessian explicitly to 
case where the usual G-N approximation is used. 
However, for usual NLS problems, the two solvers should give 
exactly the same result. This is a test for that. 
"""


def test_solver_cost_histories():
    STATE_KEY = "x"
    means = [np.array(0.5), np.array(0.9)]
    covariances = [np.eye(1), np.eye(1)]
    gaussian_mix_params = GaussianMixtureParameters(
        mixture_id="bop",
        means=means,
        covariances=covariances,
        weights=[0.5, 0.5],
    )
    component_residuals = get_component_residuals(
        gaussian_mix_params.means,
        gaussian_mix_params.covariances,
        state_key=STATE_KEY,
    )
    weights = [0.5] * 2
    res_dict = {
        "MM": MaxMixtureResidual(component_residuals, weights),
        "SM": SumMixtureResidual(component_residuals, weights),
        "MSM": MaxSumMixtureResidual(component_residuals, weights, 10),
        "HSM_STD_NO_COMPLEX": HessianSumMixtureResidualStandardCompatibility(
            component_residuals, weights, True, 0.1
        ),
    }
    solver = "LM"
    step_tol = 1e-8
    for key, res in res_dict.items():
        problem_dict = {
            "Extended": ProblemExtended(
                solver=solver,
                step_tol=step_tol,
                ftol=None,
                gradient_tol=None,
                max_iters=100,
            ),
            "Standard": Problem(
                solver=solver,
                step_tol=step_tol,
                max_iters=100,
            ),
        }
        result_list = []
        for key, problem in problem_dict.items():
            x0 = VectorState(np.array([1]), 0.0, STATE_KEY)
            problem: Problem = problem
            problem.add_residual(res)
            problem.add_variable(STATE_KEY, x0)
            result_list.append(problem.solve())
        res_ext: OptimizationSummary = result_list[0]["summary"]
        res_std: OptimizationSummary = result_list[1]["summary"]
        for lv1, cost_std_val in enumerate(res_std.entire_cost):
            cost_ext_val = res_ext.entire_cost[lv1]
            assert np.abs(cost_ext_val - cost_std_val) < 1e-2


if __name__ == "__main__":
    test_solver_cost_histories()
