"""
A set of commonly-used residuals in batch estimation. 

These residuals are
    - the PriorResidual, to assign a prior estimate on the state,
    - the ProcessResidual, which uses a navlie `ProcessModel` to compute an error between
      a predicted state and the actual state,
    - a MeasurementResidual, which uses a navlie `Measurement` to compare 
      a true measurement to the measurement predicted by the `MeasurementModel`.

"""

from abc import ABC, abstractmethod
from typing import Hashable, List, Tuple
from navlie.types import State, ProcessModel, Measurement, Input
import numpy as np


class Residual(ABC):
    """
    Abstract class for a residual to be used in batch estimation.

    Each residual must implement an evaluate(self, states) method,
    which returns an error and Jacobian of the error with
    respect to each of the states.

    Each residual must contain a list of keys, where each key corresponds to a
    variable for optimization.
    """

    jacobian_cache: List[np.ndarray] = None

    def __init__(self, keys: List[Hashable]):
        # If the hasn't supplied a list, make a list
        if isinstance(keys, list):
            self.keys = keys
        else:
            self.keys = [keys]

    @abstractmethod
    def evaluate(
        self,
        states: List[State],
        compute_jacobian: List[bool] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Evaluates the residual and Jacobians.

        Parameters
        ----------
        states : List[State]
            List of states for optimization.
        compute_jacobian : List[bool], optional
            optional flag to compute Jacobians, by default None

        Returns
        -------
        Tuple[np.ndarray, List[np.ndarray]]
            Returns the error and a list of Jacobians.
        """
        # TODO: seems more appropriate to receive states as a dict with
        # corresponding state names than as a list.
        pass

    def compute_hessians(
        self,
        states: List[State],
        compute_hessians: List[bool],
        use_jacobian_cache: bool = False,
    ) -> List[np.ndarray]:
        """Compute Hessians with respect to every state in states. In the generic case, the Hessian
        is approximated as simply jac^\trans jac. In the robustified Hessian case, this
        method will be overwritten.
        For factors that are a function of many states, the offdiagonal blocks have to be computed.
        A multidimensional array corresponding to
        [\del J \del x_1x_1 \del J \del x_1x_2
        \del J \del x_2x_1 \del J \del x_2x_2] and so on has to be returned.
        Parameters
        ----------
        states : List[State]
        compute_hessians : List[bool]
        Returns
        -------
        List[np.ndarray]
            List of Hessians for every state
        """
        hessians = [[None] * len(states) for lv1 in range(len(states))]
        if not np.array(compute_hessians).any():
            return hessians

        if not use_jacobian_cache:
            _, jac_list = self.evaluate(states, compute_jacobians=compute_hessians)
        else:
            jac_list = self.jacobian_cache.copy()
        ny = [jac.shape[0] for jac in jac_list if jac is not None][0]

        for lv1, (state, jac) in enumerate(zip(states, jac_list)):
            if jac is None:
                jac_list[lv1] = np.zeros((ny, state.dof))

        jac = np.hstack([jacobian for jacobian in jac_list])

        hessian = jac.T @ jac
        hessians = split_up_hessian_by_state(states, hessian, compute_hessians)

        return hessians

    def jacobian_fd(self, states: List[State], step_size=1e-6) -> List[np.ndarray]:
        """
        Calculates the model jacobian with finite difference.

        Parameters
        ----------
        states : List[State]
            Evaluation point of Jacobians, a list of states that
            the residual is a function of.

        Returns
        -------
        List[np.ndarray]
            A list of Jacobians of the measurement model with respect to each of the input states.
            For example, the first element of the return list is the Jacobian of the residual
            w.r.t states[0], the second element is the Jacobian of the residual w.r.t states[1], etc.
        """
        jac_list: List[np.ndarray] = [None] * len(states)

        # Compute the Jacobian for each state via finite difference
        for state_num, X_bar in enumerate(states):
            e_bar = self.evaluate(states)
            size_error = e_bar.ravel().size
            jac_fd = np.zeros((size_error, X_bar.dof))

            for i in range(X_bar.dof):
                dx = np.zeros((X_bar.dof, 1))
                dx[i, 0] = step_size
                X_temp = X_bar.plus(dx)
                state_list_pert: List[State] = []
                for state in states:
                    state_list_pert.append(state.copy())

                state_list_pert[state_num] = X_temp
                e_temp = self.evaluate(state_list_pert)
                jac_fd[:, i] = (e_temp - e_bar).flatten() / step_size

            jac_list[state_num] = jac_fd

        return jac_list

    def sqrt_info_matrix(self, states: List[State]):
        """
        Returns the information matrix
        """
        pass


def split_up_hessian_by_state(
    states: List[State], hessian: np.ndarray, compute_hessians: List[bool]
):
    hessians = [[None] * len(states) for lv1 in range(len(states))]

    slice_list = []
    start_slice = 0
    for lv1 in range(len(states)):
        end_slice = start_slice + states[lv1].dof
        slice_list.append(slice(start_slice, end_slice))
        start_slice = end_slice

    for lv1 in range(len(states)):
        for lv2 in range(len(states)):
            if not compute_hessians[lv1] or not compute_hessians[lv2]:
                hessians[lv1][lv2] = None
            else:
                hessians[lv1][lv2] = hessian[slice_list[lv1], slice_list[lv2]]

    return hessians


class PriorResidual(Residual):
    """
    A generic prior error.
    """

    def __init__(
        self,
        keys: List[Hashable],
        prior_state: State,
        prior_covariance: np.ndarray,
    ):
        super().__init__(keys)
        self._cov = prior_covariance
        self._x0 = prior_state
        # Precompute square-root of info matrix
        self._L = np.linalg.cholesky(np.linalg.inv(self._cov))

    def evaluate(
        self,
        states: List[State],
        compute_jacobians: List[bool] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Evaluates the prior error of the form

            e = x.minus(x0),

        where :math:`\mathbf{x}` is our operating point and
        :math:`\mathbf{x}_0` is a prior guess.
        """
        x = states[0]
        error = x.minus(self._x0)
        # Weight the error
        error = self._L.T @ error
        # Compute Jacobian of error w.r.t x
        if compute_jacobians:
            jacobians = [None]

            if compute_jacobians[0]:
                jacobians[0] = self._L.T @ x.minus_jacobian(self._x0)
            return error, jacobians

        return error

    def sqrt_info_matrix(self, states: List[State]):
        """
        Returns the square root of the information matrix
        """
        return self._L


class ProcessResidual(Residual):
    """
    A generic process residual.

    Can be used with any :class:`navlie.types.ProcessModel`.
    """

    def __init__(
        self,
        keys: List[Hashable],
        process_model: ProcessModel,
        u: Input,
    ):
        super().__init__(keys)
        self._process_model = process_model
        self._u = u

    def evaluate(
        self,
        states: List[State],
        compute_jacobians: List[bool] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Evaluates the process residual.

        An input :math:`\mathbf{u}` is used to propagate the state
        :math:\mathbf{x}_{k-1}` through the process model, to generate
        :math:`\hat{\mathbf{x}}_{k}`. This operation is written as

        .. math::
            \hat{\mathbf{x}}_k = \mathbf{f}(\mathbf{x}_{k-1}, \mathbf{u}, \Delta t).

        An error is then created as

            e = x_k.minus(x_k_hat),

        where :math:`\mathbf{x}_k` is our current operating point at time :math:`t_k`.
        """
        x_km1 = states[0]
        x_k = states[1]
        dt = x_k.stamp - x_km1.stamp

        # Evaluate the process model, compute the error
        x_k_hat = self._process_model.evaluate(x_km1.copy(), self._u, dt)
        e = x_k.minus(x_k_hat)

        # Scale the error by the square root of the info matrix
        L = self._process_model.sqrt_information(x_km1, self._u, dt)
        e = L.T @ e

        # Compute the Jacobians of the residual w.r.t x_km1 and x_k
        if compute_jacobians:
            jac_list = [None] * len(states)
            if compute_jacobians[0]:
                jac_list[0] = -L.T @ self._process_model.jacobian(x_km1, self._u, dt)
            if compute_jacobians[1]:
                jac_list[1] = L.T @ x_k.minus_jacobian(x_k_hat)

            return e, jac_list

        return e


class MeasurementResidual(Residual):
    """
    A generic measurement residual.

    Can be used with any :class:`navlie.Measurement`.
    """

    def __init__(self, keys: List[Hashable], measurement: Measurement):
        super().__init__(keys)
        self._y = measurement

    def evaluate(
        self,
        states: List[State],
        compute_jacobians: List[bool] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Evaluates the measurement residual.

        The error is computed as

        .. math::
            \mathbf{e} = \mathbf{y} - \mathbf{g} (\mathbf{x}).

        The Jacobian of the residual with respect to the state
        is then the negative of the measurement model Jacobian.
        """
        # Extract state
        x = states[0]

        # Compute predicted measurement
        y_check = self._y.model.evaluate(x)
        e = self._y.value.reshape((-1, 1)) - y_check.reshape((-1, 1))

        # Weight error by square root of information matrix
        L = self._y.model.sqrt_information(x)
        e = L.T @ e

        if compute_jacobians:
            jacobians = [None] * len(states)

            if compute_jacobians[0]:
                jacobians[0] = -L.T @ self._y.model.jacobian(x)
            return e, jacobians

        return e
