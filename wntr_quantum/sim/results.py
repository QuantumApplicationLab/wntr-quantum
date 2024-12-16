"""Quantum simulation results."""

from wntr.sim.results import SimulationResults


class QuantumSimulationResults(SimulationResults):
    """Quantum water network simulation results class."""

    def __init__(self):
        """Instantiate `QuantumSimulationResults`."""
        super().__init__()
        self.linear_solver_results = []

    @classmethod
    def from_simulation_results(cls, sim_results):
        """Can internally convert a standard instance of `SimulationResults` into `QuantumSimulationResults`."""
        quantum_sim_results = cls()
        # copy attributes from sim_results to new_instance
        for attr, value in sim_results.__dict__.items():
            setattr(quantum_sim_results, attr, value)
        return quantum_sim_results
