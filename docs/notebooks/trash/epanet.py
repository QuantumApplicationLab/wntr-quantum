import wntr
from quantum_newton_raphson.qubo_solver import QUBO_SOLVER
import wntr_quantum

# Create a water network model
inp_file = "networks/Net0.inp"
# inp_file = "networks/Net2Loops.inp"
wn = wntr.network.WaterNetworkModel(inp_file)

# Graph the network
wntr.graphics.plot_network(wn, title=wn.name, node_labels=True)

# define a qubo solver
linear_solver = QUBO_SOLVER(
    num_qbits=11,
    num_reads=250,
    # iterations=5,
    range=600,
    offset=250,
    # temperature=1e4,
    use_aequbols=False,
)

sim = wntr_quantum.sim.QuantumEpanetSimulator(wn, linear_solver=linear_solver)
results = sim.run_sim(linear_solver=linear_solver)

# Plot results on the network
pressure_at_5hr = results.node["pressure"].loc[0, :]
wntr.graphics.plot_network(
    wn,
    node_attribute=pressure_at_5hr,
    node_size=50,
    title="Pressure at 5 hours",
    node_labels=False,
)
