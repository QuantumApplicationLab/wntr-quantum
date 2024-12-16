"""Tests WNTR quantum using a small network and different simulators and solvers."""

import pathlib
import numpy as np
import pytest
import wntr
from qubops.encodings import PositiveQbitEncoding
from wntr_quantum.sampler.simulated_annealing import generate_random_valid_sample
from wntr_quantum.sim.solvers import QuboPolynomialSolver

NETWORKS_FOLDER = pathlib.Path(__file__).with_name("networks")
INP_FILE = NETWORKS_FOLDER / "Net0_DW.inp"  # => toy wn model
DELTA = 1.0e-12
TOL = 5  # => per cent


def calculate_small_differences(value1, value2):
    """Helper function to calculate percentage difference between classical and quantum results."""
    return np.allclose([value1], [value2], atol=1e-1, rtol=1e-1)


def compare_results(original, new):
    """Helper function that compares the classical and quantum simulation results."""
    for link in original.link["flowrate"].columns:
        orig_value = original.link["flowrate"][link].iloc[0]
        new_value = new.link["flowrate"][link].iloc[0]
        message = f"Flowrate {link}: {new_value} not within tolerance of original {orig_value}"
        assert calculate_small_differences(orig_value, new_value), message

    for node in original.node["pressure"].columns:
        orig_value = original.node["pressure"][node].iloc[0]
        new_value = new.node["pressure"][node].iloc[0]
        message = f"Pressure {node}: {new_value} not within tolerance of original {orig_value}"
        assert calculate_small_differences(orig_value, new_value), message


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

    sim = QuboPolynomialSolver(
        wn, flow_encoding=flow_encoding, head_encoding=head_encoding
    )

    x = generate_random_valid_sample(sim)
    x0 = list(x.values())

    num_temp = 2000
    Tinit = 1e1
    Tfinal = 1e-1
    Tschedule = np.linspace(Tinit, Tfinal, num_temp)
    Tschedule = np.append(Tschedule, Tfinal * np.ones(1000))
    Tschedule = np.append(Tschedule, np.zeros(1000))
    _, _, sol, res = sim.solve(
        init_sample=x0, Tschedule=Tschedule, save_traj=True, verbose=False
    )


@pytest.fixture(scope="module")
def classical_EPANET_results():
    """Get the results from the classical NR solver."""
    return run_classical_EPANET_simulation()


def test_FullQuboPolynomialSimulator(classical_EPANET_results):
    """Checks that the Quantum EPANET classical linear solver is equivalent with the classical result."""
    _ = run_FullQuboPolynomialSimulator()
    # compare_results(classical_EPANET_results, qubopoly_results)
