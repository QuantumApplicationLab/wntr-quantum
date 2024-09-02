import os
import matplotlib.pyplot as plt
import wntr
from dwave.samplers import SteepestDescentSolver
from qubols.encodings import PositiveQbitEncoding
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
inputs = ["Net0.inp"]


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

    # solve model using FULL QUBOs
    sim = wntr_quantum.sim.FullQuboPolynomialSimulator(
        wn, flow_encoding=flow_encoding, head_encoding=head_encoding
    )
    sampler = SteepestDescentSolver()
    results_quantum = sim.run_sim(solver_options={"sampler": sampler})

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
        classical_data[:2],
        quantum_data[:2],
        label="Flowrates",
        color="blue",
        marker="o",
    )
    plt.scatter(
        classical_data[2:],
        quantum_data[2:],
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
