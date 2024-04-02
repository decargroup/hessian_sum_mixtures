import pytest
import numpy as np
from navlie.lib.states import VectorState
from mixtures.gaussian_mixtures import (
    GaussianMixtureResidual,
)
from mixtures.vanilla_mixture.mixture_utils import get_components, create_residuals
import argparse


class mockArgs:
    def __init__(self, dims, weights, means, covariances):
        self.dims = dims
        self.weights = weights
        self.means = means
        self.covariances = covariances


def test_jacobians_mixtures():
    args1 = mockArgs(1, [0.5, 0.5], [0.5, 1.5], [0.5, 0.5])
    args2 = mockArgs(
        2, [0.5, 0.5], [0.5, 1.5, 1.5, 0.5], [0.5, 0, 0, 0.5, 0.2, 0, 0, 0.2]
    )
    for args in [args1, args2]:
        args.weights = [float(w) for w in args.weights]
        args.covariances = [float(w) for w in args.covariances]
        args.means = [float(w) for w in args.means]
        args.dims = int(args.dims)

        dims = args.dims

        means, covariances = get_components(args.dims, args.means, args.covariances)
        resid_dict = create_residuals(args.weights, means, covariances)
        stamp = 0.0
        n_points = 20
        test_values = n_points * np.random.rand(dims, n_points)
        test_values = [test_values[:, lv1] for lv1 in range(n_points)]

        for key in ["MM", "MSM", "SM"]:
            res: GaussianMixtureResidual = resid_dict[key]
            for x in test_values:
                test_state = VectorState(np.array([x]), stamp)
                jac_fd = res.jacobian_fd([test_state])
                err, jac_list = res.evaluate([test_state], [True])

                assert np.linalg.norm((jac_list[0] - jac_fd[0]), "fro") < 1e-5


if __name__ == "__main__":
    test_jacobians_mixtures()
