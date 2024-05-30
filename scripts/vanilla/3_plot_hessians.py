import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from mixtures.gaussian_mixtures import HessianSumMixtureResidualDirectHessian
from mixtures.vanilla_mixture.mixture_utils import (
    create_residuals,
    get_component_residuals,
    get_components,
)
from navlie.batch.gaussian_mixtures import GaussianMixtureResidual
from navlie.batch.gaussian_mixtures import (
    HessianSumMixtureResidual as HessianSumMixtureResidualStandardCompatibility,
)
from navlie.batch.gaussian_mixtures import (
    MaxMixtureResidual,
    MaxSumMixtureResidual,
    SumMixtureResidual,
)
from navlie.lib.states import VectorState

# from navlie.batch.problem import Problem


parser = argparse.ArgumentParser()

parser.add_argument(
    "--weights",
    nargs="+",
    # default=[1]
    default=[0.5, 0.5],
    # default=[1] * 5,
    type=float,
    # default=[1, 6]
    # default=[1],
)

parser.add_argument(
    "--means",
    nargs="+",
    default=[0, 0],
    # default=[0, 0, 1, -1, 2],
    type=float,
    # default=[0.5]
)

parser.add_argument(
    "--covariances",
    nargs="+",
    default=[1, 2],
    # default=[1, 2, 3, 1, 2],
    type=float,
    # default=[0.5],
)


parser.add_argument(
    "--stylesheet",
    help="Stylesheet, plots.",
    default="/home/vassili/projects/correct_sum_mixtures_/scripts/plotstylesheet_wide.mplstyle",
)

parser.add_argument(
    "--fig_name",
    help="Figure directory",
    default="/home/vassili/projects/correct_sum_mixtures_/figs/hessians_test.png",
)

parser.add_argument(
    "--plot_bounds",
    nargs="+",
    default=[-4, 4],
    type=float,
    # default=[0.5],
)


STATE_KEY = "x"


def main(args):
    sns.set_theme(style="whitegrid")
    plt.style.use(args.stylesheet)

    means, covariances = get_components(1, args.means, args.covariances)
    # Plot all the Hessians after.
    # Then point set registration example.
    res_dict = create_residuals(
        args.weights,
        means,
        covariances,
    )

    component_residuals = get_component_residuals(means, covariances)
    weights = args.weights

    res_dict = {
        "Max-Mixture": MaxMixtureResidual(component_residuals, weights),
        "Max-Sum-Mixture": MaxSumMixtureResidual(component_residuals, weights, 10),
        "Proposed": HessianSumMixtureResidualStandardCompatibility(
            component_residuals,
            weights,
            no_use_complex_numbers=True,
        ),
        "Exact": HessianSumMixtureResidualDirectHessian(
            component_residuals,
            weights,
            use_triggs=True,
            ceres_triggs_patch=False,
        ),
    }
    linestyle_dict = {
        "Max-Mixture": "--",
        "Max-Sum-Mixture": "--",
        "Exact": "-",
        "Proposed": "-",
    }
    hess_dict = {}
    x = np.linspace(args.plot_bounds[0], args.plot_bounds[1], 1000)
    for res_key, res in res_dict.items():
        res: GaussianMixtureResidual = res
        hessians = np.zeros(x.shape)
        for lv1 in range(x.shape[0]):
            state = VectorState(np.array([x[lv1]]))
            hessians[lv1] = res.compute_hessians([state], [True])[0][0].squeeze()
        hess_dict[res_key] = hessians
    cmap = sns.color_palette("colorblind")
    plt.figure()
    for lv1, res_key in enumerate(res_dict.keys()):
        plt.plot(
            x,
            hess_dict[res_key],
            label=res_key,
            color=cmap[lv1],
            linestyle=linestyle_dict[res_key],
        )
    plt.legend()
    plt.xlabel("Residual")
    plt.ylabel("Hessian")
    plt.savefig(args.fig_name, bbox_inches="tight")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
