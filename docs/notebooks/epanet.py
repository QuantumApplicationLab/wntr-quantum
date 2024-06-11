import wntr
import wntr_quantum
from quantum_newton_raphson.splu_solver import SPLU_SOLVER

# Create a water network model
inp_file = "networks/Net0.inp"
# inp_file = "networks/Net2Loops.inp"
wn = wntr.network.WaterNetworkModel(inp_file)

# Graph the network
wntr.graphics.plot_network(wn, title=wn.name, node_labels=True)

sim = wntr_quantum.sim.QuantumEpanetSimulator(wn)  # , linear_solver=SPLU_SOLVER())
results = sim.run_sim()  # linear_solver=SPLU_SOLVER())
# Plot results on the network
pressure_at_5hr = results.node["pressure"].loc[0, :]
wntr.graphics.plot_network(
    wn,
    node_attribute=pressure_at_5hr,
    node_size=50,
    title="Pressure at 5 hours",
    node_labels=False,
)
