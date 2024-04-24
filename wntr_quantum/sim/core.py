from quantum_newton_raphson.splu_solver import SPLU_SOLVER
from wntr.sim.core import WNTRSimulator
from .solvers import QuantumNewtonSolver


class QuantumWNTRSimulator(WNTRSimulator):
    """The quantum enabled NR slver."""

    def __init__(self, wn, linear_solver=SPLU_SOLVER()):  # noqa: D417
        """WNTR simulator class.
        The WNTR simulator uses a custom newton solver and linear solvers from scipy.sparse.

        Parameters
        ----------
        wn : WaterNetworkModel object
            Water network model
        linear_solver: The linear solver used for the NR step


        .. note::
            The mode parameter has been deprecated. Please set the mode using Options.hydraulic.demand_model

        """  # noqa: D205
        super(QuantumWNTRSimulator).__init__(wn)
        self._solver = QuantumNewtonSolver(linear_solver=linear_solver)
