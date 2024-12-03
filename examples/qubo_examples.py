import os
import matplotlib.pyplot as plt
import numpy as np
import wntr
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Estimator
from quantum_newton_raphson.qubo_solver import QUBO_SOLVER
from qubols.encodings import RangedEfficientEncoding
import wntr_quantum


def get_ape_from_pd_series(quantum_pd_series, classical_pd_series):
    """Helper function to evaluate absolute percentage error between classical and quantum results."""
    DELTA = 1.0e-12
    ape = (
        abs(quantum_pd_series - classical_pd_series)
        * 100.0
        / abs(classical_pd_series + DELTA)
    )
    return ape


def compare_results(classical_result, quantum_result):
    """Helper function that compares the classical and quantum simulation results."""
    TOL = 10  # => per cent
    DELTA = 1.0e-12
    classical_data = []
    quantum_data = []

    def check_ape(classical_value, quantum_value):
        """Checks if the absolute percentage error between classical and quantum results is within TOL."""
        ape = (
            abs(quantum_value - classical_value) * 100.0 / abs(classical_value + DELTA)
        )
        is_close_to_classical = ape <= TOL
        if is_close_to_classical:
            print(
                f"Quantum result {quantum_value} within {ape}% of classical result {classical_value}"
            )
            quantum_data.append(quantum_value)
            classical_data.append(classical_value)
        return is_close_to_classical

    for link in classical_result.link["flowrate"].columns:
        classical_value = classical_result.link["flowrate"][link].iloc[0]
        quantum_value = quantum_result.link["flowrate"][link].iloc[0]
        message = f"Flowrate {link}: {quantum_value} not within {TOL}% of classical result {classical_value}"
        assert check_ape(classical_value, quantum_value), message

    for node in classical_result.node["pressure"].columns:
        classical_value = classical_result.node["pressure"][node].iloc[0]
        quantum_value = quantum_result.node["pressure"][node].iloc[0]
        message = f"Pressure {node}: {quantum_value} not within {TOL}% of classical result {classical_value}"
        assert check_ape(classical_value, quantum_value), message

    return classical_data, quantum_data


# set EPANET Quantum environment variables
os.environ["EPANET_TMP"] = "/Users/murilo/scratch_dir/.epanet_quantum"
os.environ["EPANET_QUANTUM"] = "/Users/murilo/Documents/NLeSC_Projects/Vitens/EPANET"

# set input files
path = "../docs/notebooks/networks"
inputs = ["Net1Loops.inp", "Net2Loops.inp", "Net3Loops.inp"]

# set qiskit Estimator
estimator = Estimator()

for file in inputs:

    print("##################################")
    print(f"Solving for {file} model")
    print("##################################")

    # set up network model
    input_file = f"{path}/{file}"
    model_name = os.path.splitext(file)[0]
    wn = wntr.network.WaterNetworkModel(input_file)

    # solve model using the classical EPANET simulator
    sim_classical = wntr_quantum.sim.QuantumEpanetSimulator(wn)
    results_classical = sim_classical.run_sim()

    # load EPANET A and b matrices from temp
    epanet_A, epanet_b = wntr_quantum.sim.epanet.load_epanet_matrix()

    # check the size of the A and b matrices
    epanet_A_dim = epanet_A.todense().shape[0]
    epanet_b_dim = epanet_b.shape[0]
    print(f"* Size of the Jacobian in EPANET simulator: {epanet_A_dim}")
    print(f"* Size of the b vector in EPANET simulator: {epanet_b_dim}")

    # save number of nodes and pipes
    n_nodes = len(results_classical.node["pressure"].iloc[0])
    n_pipes = len(results_classical.link["flowrate"].iloc[0])

    # set number of qubits
    n_qubits = int(np.ceil(np.log2(epanet_A_dim)))

    # define ansatz
    qc = RealAmplitudes(n_qubits, reps=3, entanglement="full")

    # set the qubo solver
    qubo_solver = QUBO_SOLVER(
        encoding=RangedEfficientEncoding,
        num_qbits=15,
        num_reads=100,
        range=200,
        offset=600,
        iterations=5,
        temperature=1e4,
        use_aequbols=True,
    )

    print("* Solving model with EPANET Quantum ...\n")
    # solve model using EPANET quantum and QUBOs
    sim_quantum = wntr_quantum.sim.QuantumEpanetSimulator(wn, linear_solver=qubo_solver)
    results_quantum = sim_quantum.run_sim(linear_solver=qubo_solver)

    # plot networt and absolute percent errors
    wntr.graphics.plot_network(
        wn,
        node_attribute=get_ape_from_pd_series(
            results_quantum.node["pressure"].iloc[0],
            results_classical.node["pressure"].iloc[0],
        ),
        link_attribute=get_ape_from_pd_series(
            results_quantum.link["flowrate"].iloc[0],
            results_classical.link["flowrate"].iloc[0],
        ),
        node_colorbar_label="Pressures",
        link_colorbar_label="Flows",
        node_size=50,
        title=f"{model_name}: Absolute Percent Error",
        node_labels=False,
        filename=f"{model_name}_wnm_qubo.png",
    )

    # checks if the quantum results are within 5% of the classical ones
    classical_data, quantum_data = compare_results(results_classical, results_quantum)

    # plot all data
    plt.close()
    plt.scatter(
        classical_data[:n_pipes],
        quantum_data[:n_pipes],
        label="Flowrates",
        color="blue",
        marker="o",
    )
    plt.scatter(
        classical_data[n_pipes:],
        quantum_data[n_pipes:],
        label="Pressures",
        color="red",
        marker="s",
        facecolors="none",
    )
    plt.axline((0, 0), slope=1, linestyle="--", color="gray", label="")
    plt.xlabel("Classical EPANET results")
    plt.ylabel("Quantum EPANET results")
    plt.legend()
    plt.title(f"{model_name}")
    plt.savefig(f"{model_name}_results_qubo.png")
    plt.close()
