"""Tests WNTR quantum using a small network and different simulators and solvers."""

import pathlib
import pytest
import wntr
from dwave.samplers import SteepestDescentSolver
from qubols.encodings import PositiveQbitEncoding
import wntr_quantum

NETWORKS_FOLDER = pathlib.Path(__file__).with_name("networks")
INP_FILE = NETWORKS_FOLDER / "Net0_DW.inp"  # => toy wn model
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


def run_FullQuboPolynomialSimulator():
    """Runs QuboPolynomialSolver."""
    wn = wntr.network.WaterNetworkModel(INP_FILE)
    nqbit = 9
    step = 0.5 / (2**nqbit - 1)
    flow_encoding = PositiveQbitEncoding(
        nqbit=nqbit, step=step, offset=+1.5, var_base_name="x"
    )

    nqbit = 9
    step = 50 / (2**nqbit - 1)
    head_encoding = PositiveQbitEncoding(
        nqbit=nqbit, step=step, offset=+50.0, var_base_name="x"
    )

    sampler = SteepestDescentSolver()
    sim = wntr_quantum.sim.FullQuboPolynomialSimulator(
        wn, flow_encoding=flow_encoding, head_encoding=head_encoding
    )
    return sim.run_sim(solver_options={"sampler": sampler})


@pytest.fixture(scope="module")
def classical_EPANET_results():
    """Get the results from the classical NR solver."""
    return run_classical_EPANET_simulation()


def test_FullQuboPolynomialSimulator(classical_EPANET_results):
    """Checks that the Quantum EPANET classical linear solver is equivalent with the classical result."""
    qubopoly_results = run_FullQuboPolynomialSimulator()
    compare_results(classical_EPANET_results, qubopoly_results)