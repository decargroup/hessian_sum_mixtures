import numpy as np
import argparse
import dill as pickle
from mixtures.vanilla_mixture.monte_carlo import (
    MonteCarloRunParameters,
    OptimizationResult,
)

from navlie.batch.gaussian_mixtures import (
    GaussianMixtureResidual,
    MaxMixtureResidual,
    MaxSumMixtureResidual,
    SumMixtureResidual,
)
from mixtures.gaussian_mixtures import HessianSumMixtureResidualDirectHessian
from navlie.batch.gaussian_mixtures import (
    HessianSumMixtureResidual as HessianSumMixtureResidualStandardCompatibility,
)
from navlie.lib.states import State, VectorState
from mixtures.solver import ProblemExtended
from mixtures.vanilla_mixture.mixture_utils import (
    create_residuals,
    get_component_residuals,
)

"""
A file to rerun the optimization for a specific optimization result. Useful for debugging. 
"""

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mc_params_file",
    help="Run id from the dimension string",
    default="/home/vassili/projects/hessian_sum_mixtures/mc_results/vanilla_many_components_1d_near_step/mc_params.pkl",
)

parser.add_argument(
    "--opt_result_string",
    help="Run id from the dimension string",
    default="/home/vassili/projects/hessian_sum_mixtures/mc_results/vanilla_many_components_1d_near_step/HSM_STD_NO_COMPLEX/opt_result_mix_id_10_x0_num_30.pkl",
)

STATE_KEY = "x"


def main(args):
    with open(args.mc_params_file, "rb") as f:
        mc_params: MonteCarloRunParameters = pickle.load(f)
    with open(args.opt_result_string, "rb") as f:
        opt_result: OptimizationResult = pickle.load(f)

    gm_params = opt_result.gaussian_mix_params
    # mixture_approaches = ["MM", "SM", "MSM", "HSM", "HSM_STD", "HSM_STD_NO_COMPLEX"]
    # mixture_approaches = ["HSM", "HSM_STD_NO_COMPLEX"]
    mixture_approaches = ["HSM_STD", "HSM"]
    component_residuals = get_component_residuals(
        gm_params.means,
        gm_params.covariances,
    )
    weights = gm_params.weights
    residual_dict = get_residual_dict(component_residuals, gm_params.weights)
    x0 = opt_result.x0
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
            verbose=True,
        )
        problem.add_residual(gm_resid)
        problem.add_variable(STATE_KEY, x)
        opt_nv_res = problem.solve()


def get_residual_dict(component_residuals, weights):

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
        ),
    }

    return residual_dict


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
