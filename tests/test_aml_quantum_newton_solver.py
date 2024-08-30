"""Test QuantumNewtonSolver using a very simple aml model."""

import pytest
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Estimator
from qiskit.primitives import Sampler
from qiskit_algorithms import optimizers as opt
from quantum_newton_raphson.qubo_solver import QUBO_SOLVER
from quantum_newton_raphson.vqls_solver import VQLS_SOLVER
from qubols.encodings import EfficientEncoding
from wntr.sim import aml
from wntr_quantum.sim.quantum_newton_solver import QuantumNewtonSolver

TOL_RESULTS = 1e-2


def vqls_solver():
    """Generates a minimal VQLS linear solver."""
    estimator = Estimator()
    sampler = Sampler()
    ansatz = RealAmplitudes(num_qubits=2, entanglement="full", reps=4)
    return VQLS_SOLVER(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=opt.COBYLA(),
        sampler=sampler,
        matrix_decomposition="pauli",
        verbose=True,
    )


def qubo_solver():
    """Generates a minimal QUBO linear solver."""
    return QUBO_SOLVER(
        encoding=EfficientEncoding,
        num_qbits=15,
        num_reads=250,
        range=600,
        offset=0,
        use_aequbols=False,
    )


@pytest.mark.parametrize("linear_solver", [vqls_solver(), qubo_solver()])
def test_quantum_newton_solver(linear_solver):
    """Checks that the QuantumNewtonSolver works as expected for a simple aml model."""
    # Define a simple non-linear system of equations
    m = aml.Model()
    m.x = aml.Var(0.67)
    m.y = aml.Var(2.31)
    m.z = aml.Var(0.19)
    m.w = aml.Var(1.40)
    m.c1 = aml.Constraint(4.0 * m.x + m.y - 5.0)
    m.c2 = aml.Constraint(m.x + 3.0 * m.y + 2.0 * m.z - 8.0)
    m.c3 = aml.Constraint(2.0 * m.y + 5.0 * m.z + m.w - 7.0)
    m.c4 = aml.Constraint(m.z + 2.0 * m.w - 3.0)
    m.set_structure()

    # Solve the system
    ns = QuantumNewtonSolver(linear_solver=linear_solver)
    ns.solve(m)

    # Assertions to check the results
    assert abs(m.x.value - 0.67164) < TOL_RESULTS, "X value is incorrect"
    assert abs(m.y.value - 2.31343) < TOL_RESULTS, "Y value is incorrect"
    assert abs(m.z.value - 0.19403) < TOL_RESULTS, "Z value is incorrect"
    assert abs(m.w.value - 1.40298) < TOL_RESULTS, "W value is incorrect"

    # Check the residuals if relevant
    # residuals = m.evaluate_residuals()
    # for res in residuals:
    #     assert abs(res) < TOL_RESULTS, f"Residual too high: {res}"
