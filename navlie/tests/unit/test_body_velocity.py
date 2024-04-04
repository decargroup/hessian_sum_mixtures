from navlie.lib import SE2State, SE3State
from navlie.lib import (
    BodyFrameVelocity,
    RelativeBodyFrameVelocity,
    VectorInput,
)
from pymlg import SE3, SE2, SE3
import numpy as np
import pytest


@pytest.mark.parametrize("direction", ["left", "right"])
def test_body_velocity_se3(direction):
    x = SE3State(
        SE3.random(),
        direction=direction,
    )
    u = VectorInput(np.array([1, 2, 3, 4, 5, 6]))
    dt = 0.1
    Q = np.identity(6)
    process_model = BodyFrameVelocity(Q)
    jac = process_model.jacobian(x, u, dt)
    jac_fd = process_model.jacobian_fd(x, u, dt)
    assert np.allclose(jac, jac_fd, atol=1e-4)


@pytest.mark.parametrize("direction", ["left", "right"])
def test_body_velocity_se2(direction):
    x = SE2State(
        SE2.random(),
        direction=direction,
    )
    u = VectorInput(np.array([1, 2, 3]))
    dt = 0.1
    Q = np.identity(3)
    process_model = BodyFrameVelocity(Q)
    jac = process_model.jacobian(x, u, dt)
    jac_fd = process_model.jacobian_fd(x, u, dt)
    Q = process_model.covariance(x, u, dt)
    assert np.allclose(jac, jac_fd, atol=1e-4)


@pytest.mark.parametrize("direction", ["left", "right"])
def test_body_velocity_se3_left(direction):
    x = SE3State(
        SE3.random(),
        direction=direction,
    )
    u = VectorInput(np.array([1, 2, 3, 4, 5, 6]))
    dt = 0.1
    Q = np.identity(6)
    process_model = BodyFrameVelocity(Q)
    jac = process_model.jacobian(x, u, dt)
    jac_fd = process_model.jacobian_fd(x, u, dt)
    assert np.allclose(jac, jac_fd, atol=1e-4)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_body_velocity_se2_left(direction):
    x = SE2State(
        SE2.random(),
        direction=direction,
    )
    u = VectorInput(np.array([1, 2, 3]))
    dt = 0.1
    Q = np.identity(3)
    process_model = BodyFrameVelocity(Q)
    jac = process_model.jacobian(x, u, dt)
    jac_fd = process_model.jacobian_fd(x, u, dt)
    Q = process_model.covariance(x, u, dt)
    assert np.allclose(jac, jac_fd, atol=1e-4)


def test_relative_body_velocity_se2():
    x = SE2State(
        SE2.random(),
        direction="right",
    )
    u = VectorInput(np.array([1, 2, 3, 4, 5, 6]))
    dt = 0.1
    Q = np.identity(3)
    process_model = RelativeBodyFrameVelocity(Q, Q)
    jac = process_model.jacobian(x, u, dt)
    jac_fd = process_model.jacobian_fd(x, u, dt)
    Q = process_model.covariance(x, u, dt)
    assert np.allclose(jac, jac_fd, atol=1e-4)


def test_relative_body_velocity_se3():
    x = SE3State(
        SE3.random(),
        direction="right",
    )
    u = VectorInput(np.array([i for i in range(12)]))
    dt = 0.1
    Q = np.identity(6)
    process_model = RelativeBodyFrameVelocity(Q, Q)
    jac = process_model.jacobian(x, u, dt)
    jac_fd = process_model.jacobian_fd(x, u, dt)
    Q = process_model.covariance(x, u, dt)
    assert np.allclose(jac, jac_fd, atol=1e-4)


def test_relative_body_velocity_equivalence():
    T_a1 = SE3.random()
    T_a2 = SE3.random()
    T_12 = SE3.inverse(T_a1) @ T_a2

    x = SE3State(
        T_12,
        direction="right",
    )
    u = VectorInput(np.array([i for i in range(12)]).reshape((-1, 6)))
    dt = 0.1
    Q = np.identity(6)
    process_model = RelativeBodyFrameVelocity(Q, Q)
    T_12_k = process_model.evaluate(x, u, dt)
    T_a1_k = T_a1 @ SE3.Exp(u.value[0] * dt)
    T_a2_k = T_a2 @ SE3.Exp(u.value[1] * dt)
    T_12_k_test = SE3.inverse(T_a1_k) @ T_a2_k

    assert np.allclose(T_12_k.value, T_12_k_test, atol=1e-12)


if __name__ == "__main__":
    test_body_velocity_se3_left()
