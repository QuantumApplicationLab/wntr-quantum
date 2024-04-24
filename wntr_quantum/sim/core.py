from wntr.sim.core import WNTRSimulator 
from .solvers import QuantumNewtonSolver
from quantum_newton_raphson.splu_solver import SPLU_SOLVER

class QuantumWNTRSimulator(WNTRSimulator):
    
    def __init__(self, wn, linear_solver=SPLU_SOLVER()):
        """
        WNTR simulator class.
        The WNTR simulator uses a custom newton solver and linear solvers from scipy.sparse.

        Parameters
        ----------
        wn : WaterNetworkModel object
            Water network model


        .. note::
        
            The mode parameter has been deprecated. Please set the mode using Options.hydraulic.demand_model

        """

        super(QuantumWNTRSimulator).__init__(wn)
        self._solver = QuantumNewtonSolver(linear_solver=linear_solver)