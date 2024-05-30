from typing import List, Tuple

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal

import mixtures
from navlie.batch.gaussian_mixtures import GaussianMixtureResidual
from navlie.lib.states import VectorState


def get_plot_bounds(means: List[np.ndarray], covariances: List[np.ndarray]):
    dims = means[0].shape[0]
    ymin = None
    ymax = None
    if dims == 1:
        xmin = min([mean - 3 * np.sqrt(cov) for mean, cov in zip(means, covariances)])
        xmax = max([mean + 3 * np.sqrt(cov) for mean, cov in zip(means, covariances)])
    if dims == 2:
        xmin = min(
            [mean[0] - 3 * np.sqrt(cov[0, 0]) for mean, cov in zip(means, covariances)]
        )
        xmax = max(
            [mean[0] + 3 * np.sqrt(cov[0, 0]) for mean, cov in zip(means, covariances)]
        )
        ymin = min(
            [mean[1] - 3 * np.sqrt(cov[1, 1]) for mean, cov in zip(means, covariances)]
        )
        ymax = max(
            [mean[1] + 3 * np.sqrt(cov[1, 1]) for mean, cov in zip(means, covariances)]
        )
    return xmin, xmax, ymin, ymax


def get_bounds_from_args(args):
    xmin = float(args.xmin)
    xmax = float(args.xmax)
    if args.ymin is None:
        ymin = xmin
        ymax = xmax
    else:
        ymin = float(args.ymin)
        ymax = float(args.ymax)

    return xmin, xmax, ymin, ymax


def plot_residuals_1d(
    ax: plt.Axes,
    mm_res: GaussianMixtureResidual,
    sm_res: GaussianMixtureResidual,
    msm_res: GaussianMixtureResidual,
    xmin: float,
    xmax: float,
    cmap: List[int] = None,
):
    key = mm_res.keys[0]
    if cmap is None:
        cmap = get_plot_colormap(3)

    x_arr = np.linspace(xmin, xmax, 1000).squeeze()
    # print(np.linalg.norm(mm_res.evaluate([VectorState(x, 0.0, key)])) ** 2)
    log_max_mix = [
        np.linalg.norm(mm_res.evaluate([VectorState(x, 0.0, key)])) ** 2
        for x in x_arr.tolist()
    ]
    log_sm_mix = [
        np.linalg.norm(sm_res.evaluate([VectorState(x, 0.0, key)])) ** 2
        for x in x_arr.tolist()
    ]
    log_msm_mix = [
        np.linalg.norm(msm_res.evaluate([VectorState(x, 0.0, key)])) ** 2
        for x in x_arr.tolist()
    ]

    ax.plot(x_arr, log_max_mix, label="Max Mixture Residual", color=cmap[0])
    ax.plot(x_arr, log_sm_mix, label="Sum Mixture Residual", color=cmap[1])
    ax.plot(x_arr, log_msm_mix, label="Max Sum Mixture Residual", color=cmap[2])


def plot_mixture_1d(
    ax: plt.Axes,
    weights: List[float],
    means: List[np.ndarray],
    covariances: List[np.ndarray],
    plot_components: bool = False,
    plot_max_mixture: bool = False,
    plot_full_mixture: bool = False,
    plot_likelihoods: bool = False,
    cmap: List[int] = None,
    xmin: int = None,
    xmax: int = None,
):
    if cmap is None:
        cmap = get_plot_colormap(3)
    x = np.linspace(xmin, xmax, 1000)
    x = x.squeeze()
    mix_list = []
    unweighted_mix_list = []
    for weight, mean, cov in zip(weights, means, covariances):
        mix_list.append(
            weight * multivariate_normal.pdf(x, mean=mean, cov=cov).reshape(1, -1)
        )
        unweighted_mix_list.append(
            multivariate_normal.pdf(x, mean=mean, cov=cov).reshape(1, -1)
        )

    mix = np.sum(np.concatenate(mix_list, axis=0), axis=0)

    max_mix_indices = np.argmax(np.concatenate(mix_list, axis=0), axis=0)
    max_mix = []
    for lv1, idx in enumerate(max_mix_indices):
        # idx corresponds to max idx of mixture
        # lv1 is time index
        weight = weights[idx]
        max_mix.append(weight * unweighted_mix_list[idx][0, lv1])
    max_mix = np.array(max_mix)

    if plot_full_mixture:
        ax.plot(x, mix, label="Full Mixture", color=cmap[1])
    if plot_max_mixture:
        ax.plot(x, max_mix, "--", label="Max Mixture", color=cmap[0])

    if plot_components:
        for lv1 in range(len(mix_list)):
            ax.plot(
                x.squeeze(),
                mix_list[lv1].squeeze(),
                label=f"Weight: {weights[lv1]}, Mean: {means[lv1]}, Cov: {covariances[lv1]}",
            )
    if plot_likelihoods:
        ax.plot(x, -np.log(mix), label="Full Mixture Likelihood")
        ax.plot(x, -np.log(max_mix), label="Max Mixture Likelihood")


def plot_mixture_2d(
    ax: plt.Axes,
    weights: List[float],
    means: List[np.ndarray],
    covariances: List[np.ndarray],
    n_points_per_axis: int = 20,
    plot_max_mixture: bool = False,
    plot_full_mixture: bool = False,
    plot_likelihood_full: bool = False,
    plot_likelihood_max: bool = False,
    plot_3d_mix: bool = False,
    xlim: List[float] = None,
    ylim: List[float] = None,
    cmap="viridis",
    return_max=True,
    levels: List[float] = None,
):
    if xlim is None:
        xmin = min(
            [mean[0] - 3 * np.sqrt(cov[0, 0]) for mean, cov in zip(means, covariances)]
        )
        xmax = max(
            [mean[0] + 3 * np.sqrt(cov[0, 0]) for mean, cov in zip(means, covariances)]
        )
    else:
        xmin = xlim[0]
        xmax = xlim[1]
    if ylim is None:
        ymin = min(
            [mean[1] - 3 * np.sqrt(cov[1, 1]) for mean, cov in zip(means, covariances)]
        )
        ymax = max(
            [mean[1] + 3 * np.sqrt(cov[1, 1]) for mean, cov in zip(means, covariances)]
        )
    else:
        ymin = ylim[0]
        ymax = ylim[1]
    # print(f"Xmin: {xmin}, Xmax: {xmax}, Ymin: {ymin}, Ymax: {ymax}")
    # print("Means:")
    # print(means)
    # print("Covariances")
    # print(covariances)
    mix_list = []
    unweighted_mix_list = []
    x = np.linspace(xmin, xmax, n_points_per_axis)
    y = np.linspace(ymin, ymax, n_points_per_axis)
    X, Y = np.meshgrid(x, y)  # grid of point

    for weight, mean, cov in zip(weights, means, covariances):
        Z = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = multivariate_normal.pdf(
                    np.array([X[i, j], Y[i, j]]), mean=mean, cov=cov
                )
        mix_list.append(weight * np.expand_dims(Z, axis=2))
        unweighted_mix_list.append(Z)

    mix = np.sum(np.concatenate(mix_list, axis=2), axis=2)

    max_mix_indices = np.argmax(np.concatenate(mix_list, axis=2), axis=2)
    max_mix = np.zeros(X.shape)
    for lv1 in range(max_mix_indices.shape[0]):
        for lv2 in range(max_mix_indices.shape[1]):
            # idx corresponds to max idx of mixture
            # lv1 is time index
            idx = max_mix_indices[lv1, lv2]
            weight = weights[idx]
            max_mix[lv1, lv2] = weight * unweighted_mix_list[idx][lv1, lv2]

    if plot_full_mixture:
        cs = ax.contourf(X, Y, mix, label="Full Mixture", cmap=cmap, levels=levels)
    if plot_max_mixture:
        cs = ax.contourf(X, Y, max_mix, label="Max Mixture", cmap=cmap, levels=levels)
    if plot_likelihood_full:
        cs = ax.contourf(
            X,
            Y,
            -np.log(mix),
            label="Full Mixture Log Likelihood",
            cmap=cmap,
            levels=levels,
        )
    if plot_likelihood_max:
        cs = ax.contourf(
            X,
            Y,
            -np.log(max_mix),
            label="Max Mixture Likelihood",
            cmap=cmap,
            levels=levels,
        )
    if plot_3d_mix:
        ax.plot_surface(X, Y, mix, cmap=cmap)
    if return_max:
        # Get argmax of probability
        argmax_2d = np.unravel_index(np.argmax(mix, axis=None), mix.shape)
        return np.array([X[argmax_2d], Y[argmax_2d]]), cs


def plot_residual_2d(
    ax: plt.Axes,
    gm_res: GaussianMixtureResidual,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    n_points_per_axis: int,
    label: str = None,
    fig: plt.Figure = None,
    cmap="viridis",
):
    x = np.linspace(xmin, xmax, n_points_per_axis)
    y = np.linspace(ymin, ymax, n_points_per_axis)
    X, Y = np.meshgrid(x, y)  # grid of point
    Z = np.zeros(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            state = VectorState(np.array([X[i, j], Y[i, j]]))
            Z[i, j] = np.linalg.norm(gm_res.evaluate([state])) ** 2
    CS = ax.contourf(X, Y, Z, label=label, cmap=cmap)
    if fig is not None:
        cbar = fig.colorbar(CS)


def get_plot_colormap(n: int, name: str = "Dark2"):
    interval = [lv1 for lv1 in range(n)]
    cmap = cm.get_cmap(name)
    colors = [cmap(x) for x in interval]
    return colors


def plot_iterate_lists(
    ax: plt.Axes,
    labels: List[str],
    iterate_lists: List[List[List[np.ndarray]]],
    true_x: VectorState,
    cmap: List[str],
    log_scale: bool = False,
):
    for lv2, (label, iterate_history_list_mixture) in enumerate(
        zip(labels, iterate_lists)
    ):
        for lv1, iterate_history in enumerate(iterate_history_list_mixture):
            label = None
            if lv1 == 0:
                label = labels[lv2]
            distance = np.array([np.linalg.norm(x - true_x) for x in iterate_history])

            if not np.isnan(distance).any():
                if log_scale:
                    distance = np.log(distance)
                ax.plot(
                    distance,
                    color=cmap[lv2],
                    label=label,
                )


def scatter_plot_optimization_result(
    ax: plt.Axes,
    x_list: List[VectorState],
    res: GaussianMixtureResidual,
    color: Tuple[int],
    label=None,
    markerType=None,
    markerSize=None,
):
    dims = x_list[0].value.shape[0]
    if dims == 1:
        ax.scatter(
            np.array([x.value for x in x_list]).squeeze(),
            np.array(list([np.linalg.norm(res.evaluate([x])) ** 2] for x in x_list)),
            color=color,
            label=label,
        )
    if dims == 2:
        ax.scatter(
            np.array([x.value[0] for x in x_list]).squeeze(),
            np.array([x.value[1] for x in x_list]).squeeze(),
            color=color,
            label=label,
            s=markerSize,
            marker=markerType,
        )
