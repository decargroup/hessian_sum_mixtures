import numpy as np
import pytest
from pymlg import SE2, SE3

from mixtures.point_set_registration.point_set_registration import (
    SinglePointPsrResidual,
    SinglePointPsrResidual3d,
)
from navlie.lib.states import SE2State, SE3State


def test_psr_jacobians_2d():
    key = "x"
    source_landmark = np.random.uniform(low=-1, high=1, size=(2,))
    ref_landmark = np.random.uniform(low=-1, high=1, size=(2,))
    source_cov = np.diag(np.random.uniform(low=0.5, high=7, size=(2,)))
    ref_cov = np.diag(np.random.uniform(low=0.5, high=7, size=(2,)))
    x_state = SE2State(value=SE2.random(), direction="left")
    psr_res = SinglePointPsrResidual(
        [key],
        source_landmark,
        ref_landmark,
        source_cov,
        ref_cov,
    )
    jac_fd = psr_res.jacobian_fd([x_state])[0]
    e, jac_analytic_list = psr_res.evaluate([x_state], [True])
    jac_analytic = jac_analytic_list[0]
    # print(pd.DataFrame(jac_fd))
    # print(pd.DataFrame(jac_analytic))
    assert np.linalg.norm(jac_fd[:, 1:] - jac_analytic[:, 1:], "fro") < 1e-6

    # This one will never quiiiteee match because we neglect the
    # dependence of the error covariance on the rotation in
    # deriving the analytic Jacobian
    assert np.linalg.norm(jac_fd[0, :] - jac_analytic[0, :], 2) < 1


def test_psr_jacobians_3d():
    key = "x"
    source_landmark = np.random.uniform(low=-1, high=1, size=(3,))
    ref_landmark = np.random.uniform(low=-1, high=1, size=(3,))
    source_cov = np.diag(np.random.uniform(low=0.5, high=7, size=(3,)))
    ref_cov = np.diag(np.random.uniform(low=0.5, high=7, size=(3,)))
    x_state = SE3State(value=SE3.random(), direction="left")
    psr_res = SinglePointPsrResidual3d(
        [key],
        source_landmark,
        ref_landmark,
        source_cov,
        ref_cov,
    )
    jac_fd = psr_res.jacobian_fd([x_state])[0]
    e, jac_analytic_list = psr_res.evaluate([x_state], [True])
    jac_analytic = jac_analytic_list[0]
    # print(pd.DataFrame(jac_fd))
    # print(pd.DataFrame(jac_analytic))
    assert np.linalg.norm(jac_fd[:, 3:] - jac_analytic[:, 3:], "fro") < 1e-6

    # Same as 2d case, this one will never quiiiteee match because we neglect the
    # dependence of the error covariance on the rotation in
    # deriving the analytic Jacobian
    assert np.linalg.norm(jac_fd[:3, :] - jac_analytic[:3, :], 2) < 1


if __name__ == "__main__":
    test_psr_jacobians_2d()
    test_psr_jacobians_3d()
