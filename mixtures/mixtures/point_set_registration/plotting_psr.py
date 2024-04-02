from matplotlib import pyplot as plt
import numpy as np
from typing import List
from pymlg import SO2
import matplotlib as mpl

# For arrows in 3D: https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D


class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    """Add an 3d arrow to an `Axes3D` instance."""

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, "arrow3D", _arrow3D)


def draw_reference_frame(
    ax: plt.Axes,
    color: str,
    origin: np.ndarray,
    C_ab: np.ndarray,
    arrow_length: float,
    head_width=0.5,
    head_length=0.75,
):
    dims = origin.shape[0]
    if dims == 2:
        ax.arrow(
            origin[0],
            origin[1],
            dx=arrow_length * C_ab[0, 0],
            dy=arrow_length * C_ab[1, 0],
            color=color,
            head_width=head_width,
            head_length=head_length,
        )
        ax.arrow(
            origin[0],
            origin[1],
            dx=arrow_length * C_ab[0, 1],
            dy=arrow_length * C_ab[1, 1],
            color=color,
            head_width=head_width,
            head_length=head_length,
        )
    if dims == 3:
        ax.arrow3D(
            origin[0],
            origin[1],
            origin[2],
            origin[0] + arrow_length * C_ab[0, 0],
            origin[1] + arrow_length * C_ab[1, 0],
            origin[2] + arrow_length * C_ab[2, 0],
            color=color,
            mutation_scale=20,
        )
        ax.arrow3D(
            origin[0],
            origin[1],
            origin[2],
            origin[0] + arrow_length * C_ab[0, 1],
            origin[1] + arrow_length * C_ab[1, 1],
            origin[2] + arrow_length * C_ab[2, 1],
            color=color,
            mutation_scale=20,
        )
        ax.arrow3D(
            origin[0],
            origin[1],
            origin[2],
            origin[0] + arrow_length * C_ab[0, 2],
            origin[1] + arrow_length * C_ab[1, 2],
            origin[2] + arrow_length * C_ab[2, 2],
            color=color,
            mutation_scale=20,
        )


#
# ax.arrow(
#     origin[0],
#     origin[1],
#     dx=arrow_length * np.cos(angle),
#     dy=arrow_length * np.sin(angle),
#     color=color,
#     head_width=head_width,
#     head_length=head_length,
# )
# ax.arrow(
#     origin[0],
#     origin[1],
#     dx=arrow_length * -np.sin(angle),
#     dy=arrow_length * np.cos(angle),
#     color=color,
#     head_width=head_width,
#     head_length=head_length,
# )


def ellipsoid_from_covariance(P: np.ndarray):
    """Get 1sigma ellipsoid parameters from corresponding covariance

    Parameters
    ----------
    P : np.ndarray
        Covariance matrix

    Returns
    -------
    width, height, angle : float
        Ellipsoid parameters, suitable for matplotlib Ellipse
        Angle is in radians
    """
    # Eigendecomposition of P
    eigvals, eigvecs = np.linalg.eig(P)

    width = np.sqrt(eigvals[0]) * 2
    height = np.sqrt(eigvals[1]) * 2
    angle = np.arctan2(eigvecs[0, 1], eigvecs[0, 0])
    return width, height, angle


def plot_problem_setup(
    ax: plt.Axes,
    ref_landmarks: np.ndarray,
    source_landmarks: np.ndarray,
    ref_covs: List[np.ndarray],
    source_covs: List[np.ndarray],
    bounds: List[float],
    color_ref="black",
    color_source="red",
):
    ax.scatter(
        ref_landmarks[0, :],
        ref_landmarks[1, :],
        facecolors="none",
        edgecolors=color_ref,
        label="Reference",
    )

    ax.scatter(
        source_landmarks[0, :],
        source_landmarks[1, :],
        facecolors="none",
        edgecolors=color_source,
        label="Source",
    )

    ax.set_xlim(bounds)
    ax.set_ylim(bounds)

    # Plot correspondence lines
    for ref_landmark, source_landmark in zip(ref_landmarks.T, source_landmarks.T):
        ax.plot(
            [ref_landmark[0], source_landmark[0]],
            [ref_landmark[1], source_landmark[1]],
            color=color_ref,
            alpha=0.5,
        )
    for lv1 in range(len(ref_covs)):
        ref_landmark = ref_landmarks[:, lv1]
        ref_cov = ref_covs[lv1]
        source_landmark = source_landmarks[:, lv1]
        source_cov = source_covs[lv1]

        width, height, angle_rad = ellipsoid_from_covariance(ref_cov)
        ellipse = mpl.patches.Ellipse(
            xy=ref_landmark,
            width=width,
            height=height,
            angle=angle_rad * 180 / np.pi,
            # angle=-45,
            alpha=0.5,
            fill=False,
            edgecolor=color_ref,
        )
        ax.add_patch(ellipse)
        width, height, angle_rad = ellipsoid_from_covariance(source_cov)
        # print(f"Source angle {angle_rad * 180 / np.pi}")
        ellipse = mpl.patches.Ellipse(
            xy=source_landmark,
            width=width,
            height=height,
            angle=angle_rad * 180 / np.pi,
            fill=False,
            edgecolor=color_source,
        )
        ax.add_patch(ellipse)
        # ax.annotate(f"{angle_rad * 180 / np.pi:.2f}", source_landmark, size=10)
