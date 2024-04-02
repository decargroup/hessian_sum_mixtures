import mixtures
import argparse
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import scipy
from scipy.stats import multivariate_normal
import seaborn as sns

from navlie.batch.residuals import PriorResidual
from navlie.lib.states import VectorState
from mixtures.gaussian_mixtures import (
    MaxMixtureResidual,
    SumMixtureResidual,
    MaxSumMixtureResidual,
    HessianSumMixtureResidual,
    HessianSumMixtureResidualStandardCompatibility,
)

from typing import Tuple


def get_components(dims: int, means: List[float], covariances: List[float]):
    n_components = int(len(means) / dims)
    means = [
        np.array(means[lv1 * dims : (1 + lv1) * dims]) for lv1 in range(n_components)
    ]
    covariances = [
        np.array(covariances[lv1 * dims**2 : (1 + lv1) * dims**2]).reshape(dims, dims)
        for lv1 in range(n_components)
    ]
    return means, covariances


def generate_initial_guesses(
    dims: int,
    means: List[float],
    covariances: List[float],
    n_guesses: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> List[np.ndarray]:
    x_list = []
    if dims == 1:
        x_arr = np.linspace(xmin, xmax, n_guesses)
        x_list = x_arr.tolist()
    if dims == 2:
        x_list = []
        n_points_per_axis = int(np.floor(np.sqrt(n_guesses)))
        x = np.linspace(xmin, xmax, n_points_per_axis)
        y = np.linspace(ymin, ymax, n_points_per_axis)
        X, Y = np.meshgrid(x, y)  # grid of point
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x_list.append(np.array([X[i, j], Y[i, j]]))

    return x_list


def get_component_residuals(
    means: List[np.ndarray],
    covariances: List[np.ndarray],
    state_key: str = "x",
):
    key = state_key
    component_residuals = []
    stamp = 0.0
    dims = covariances[0].shape[0]
    for lv1 in range(len(means)):
        prior_state = VectorState(means[lv1], stamp)
        prior_covariance = np.array(covariances[lv1]).reshape(dims, dims)
        component_residuals.append(PriorResidual([key], prior_state, prior_covariance))
    return component_residuals


def create_residuals(
    weights: List[float],
    means: List[np.ndarray],
    covariances: List[np.ndarray],
    state_key: str = "x",
    msm_damping_const: float = 10,
):
    key = state_key
    component_residuals = []
    stamp = 0.0
    dims = covariances[0].shape[0]
    for lv1 in range(len(weights)):
        prior_state = VectorState(means[lv1], stamp)
        prior_covariance = np.array(covariances[lv1]).reshape(dims, dims)
        component_residuals.append(PriorResidual([key], prior_state, prior_covariance))

    res_dict = {
        "MM": MaxMixtureResidual(component_residuals, weights),
        "SM": SumMixtureResidual(component_residuals, weights),
        "MSM": MaxSumMixtureResidual(component_residuals, weights, msm_damping_const),
        "HSM": HessianSumMixtureResidual(
            component_residuals,
            weights,
            use_triggs=False,
            ceres_triggs_patch=False,
        ),
        "HSM_EXACT": HessianSumMixtureResidual(
            component_residuals,
            weights,
            use_triggs=True,
            ceres_triggs_patch=False,
        ),
        "HSM_STD": HessianSumMixtureResidualStandardCompatibility(
            component_residuals, weights, no_use_complex_numbers=False
        ),
        "HSM_NO_COMPLEX": HessianSumMixtureResidualStandardCompatibility(
            component_residuals, weights, no_use_complex_numbers=True
        ),
    }

    return res_dict


def decompose_result_list(
    result_list,
) -> Tuple[List[VectorState], List[List[float]], List[List[VectorState]]]:
    x_list = [result["variables"]["x"] for result in result_list]
    cost_list = [result["summary"].entire_cost for result in result_list]
    iterate_history_list = [result["summary"].iterate_history for result in result_list]
    for lv1 in range(len(iterate_history_list)):
        iterate_history_list[lv1] = [
            iterate_states[0]["x"].value.reshape(-1, 1)
            for iterate_states in iterate_history_list[lv1]
        ]
    return x_list, cost_list, iterate_history_list
