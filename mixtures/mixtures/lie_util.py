from navlie.types import State
import numpy as np


def minus_jacobian(Y: State, X: State, arg="Y") -> np.ndarray:
    """Yields the Lie Jacobian of the minus operation. The argument with respect to which
    the Jacobian is desired is specified by arg parameter.
    \mbs{\tau} = \mbs{Y}\ominus \mbs{X}

    Parameters
    ----------
    Y : State
    X : State
    arg : str, optional
        the argument of the minus operation with respect to which the Jacobian is desired, by default "Y".

    Returns
    -------
    np.ndarray
        Jacobian of the minus operation, square with size Y.dof
    """
    if arg == "Y":
        return np.linalg.inv(X.plus_jacobian(Y.minus(X)))
    if arg == "X":
        return -np.linalg.inv(X.plus_jacobian(-Y.minus(X)))
