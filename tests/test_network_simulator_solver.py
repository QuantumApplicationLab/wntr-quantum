"""Tests WNTR quantum using a small network and different simulators and solvers."""

import pathlib
import pytest
import wntr
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Estimator
from qiskit_algorithms.optimizers import CG
from quantum_newton_raphson.qubo_solver import QUBO_SOLVER
from quantum_newton_raphson.vqls_solver import VQLS_SOLVER
import wntr_quantum
from qubops.encodings import PositiveQbitEncoding

NETWORKS_FOLDER = pathlib.Path(__file__).with_name("networks")
INP_FILE = NETWORKS_FOLDER / "Net0.inp"  # => toy wn model
DELTA = 1.0e-12
TOL = 5  # => per cent


def calculate_differences(value1, value2):
    """Helper function to calculate percentage difference between classical and quantum results."""
    return abs(value1 - value2) / abs(value1 + DELTA) <= TOL / 100.0


def compare_results(original, new):
    """Helper function that compares the classical and quantum simulation results."""
    for link in original.link["flowrate"].columns:
        orig_value = original.link["flowrate"][link].iloc[0]
        new_value = new.link["flowrate"][link].iloc[0]
        message = (
            f"Flowrate {link}: {new_value} not within {TOL}% of original {orig_value}"
        )
        assert calculate_differences(orig_value, new_value), message

    for node in original.node["pressure"].columns:
        orig_value = original.node["pressure"][node].iloc[0]
        new_value = new.node["pressure"][node].iloc[0]
        message = (
            f"Pressure {node}: {new_value} not within {TOL}% of original {orig_value}"
        )
        assert calculate_differences(orig_value, new_value), message


def run_classical_EPANET_simulation():
    """Runs WNTR using classical EPANET interface."""
    wn = wntr.network.WaterNetworkModel(INP_FILE)
    sim = wntr.sim.EpanetSimulator(wn)
    return sim.run_sim()


def run_classical_WNTR_simulation():
    """Runs WNTR using its classical simulator."""
    wn = wntr.network.WaterNetworkModel(INP_FILE)
    sim = wntr.sim.WNTRSimulator(wn)
    return sim.run_sim()


def run_QuantumEpanetSimulator_with_cholesky():
    """Runs QuantumEpanetSimulator with default CholeskySolver."""
    wn = wntr.network.WaterNetworkModel(INP_FILE)
    sim = wntr_quantum.sim.QuantumEpanetSimulator(wn)
    return sim.run_sim()


def run_QuantumWNTRSimulator_with_splu():
    """Runs QuantumWNTRSimulator with default SPLU_SOLVER."""
    wn = wntr.network.WaterNetworkModel(INP_FILE)
    sim = wntr_quantum.sim.QuantumWNTRSimulator(wn)
    return sim.run_sim()


def run_QuantumEpanetSimulator_with_qubols(use_aequbols):
    """Runs QuantumEpanetSimulator with QUBOLS solver."""
    wn = wntr.network.WaterNetworkModel(INP_FILE)
    linear_solver = QUBO_SOLVER(
        num_qbits=11,
        num_reads=250,
        range=600,
        use_aequbols=use_aequbols,
    )
    sim = wntr_quantum.sim.QuantumEpanetSimulator(wn, linear_solver=linear_solver)
    return sim.run_sim(linear_solver=linear_solver)


def run_QuantumEpanetSimulator_with_vqls():
    """Runs QuantumEpanetSimulator with VQLS solver."""
    wn = wntr.network.WaterNetworkModel(INP_FILE)
    qc = RealAmplitudes(1, reps=3, entanglement="full")
    estimator = Estimator()
    linear_solver = VQLS_SOLVER(
        estimator=estimator, ansatz=qc, optimizer=CG(), matrix_decomposition="pauli"
    )
    sim = wntr_quantum.sim.QuantumEpanetSimulator(wn, linear_solver=linear_solver)
    return sim.run_sim(linear_solver=linear_solver)


def run_FullQuboPolynomialSimulator():
    """"""


@pytest.fixture(scope="module")
def classical_EPANET_results():
    """Get the results from the classical NR solver."""
    return run_classical_EPANET_simulation()


def test_QuantumEpanetSimulator_cholesky(classical_EPANET_results):
    """Checks that the Quantum EPANET classical linear solver is equivalent with the classical result."""
    cholesky_classical_results = run_QuantumEpanetSimulator_with_cholesky()
    compare_results(classical_EPANET_results, cholesky_classical_results)


@pytest.mark.parametrize("use_aequbols", [True, False])
def test_QuantumEpanetSimulator_qubols_solver(classical_EPANET_results, use_aequbols):
    """Checks that the Quantum EPANET QUBOLS solver works as expected."""
    qubols_quantum_results = run_QuantumEpanetSimulator_with_qubols(use_aequbols)
    compare_results(classical_EPANET_results, qubols_quantum_results)


def test_QuantumEpanetSimulator_vqls_solver(classical_EPANET_results):
    """Checks that the Quantum EPANET VQLS solver works as expected."""
    vqls_quantum_results = run_QuantumEpanetSimulator_with_vqls()
    compare_results(classical_EPANET_results, vqls_quantum_results)


@pytest.fixture(scope="module")
def classical_WNTR_results():
    """Get the results from the classical NR solver from WNTR."""
    return run_classical_WNTR_simulation()


def test_QuantumWNTRSimulator_splu(classical_WNTR_results):
    """Checks that the QuantumWNTRSimulator with splu linear solver is equivalent with the classical result."""
    splu_classical_results = run_QuantumWNTRSimulator_with_splu()
    compare_results(classical_WNTR_results, splu_classical_results)
