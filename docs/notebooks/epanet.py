import wntr
import wntr_quantum

# Create a water network model
inp_file = "networks/Net0.inp"
wn = wntr.network.WaterNetworkModel(inp_file)

# Graph the network
wntr.graphics.plot_network(wn, title=wn.name, node_labels=True)

sim = wntr_quantum.sim.QuantumEpanetSimulator(wn)
results = sim.run_sim()
# Plot results on the network
pressure_at_5hr = results.node["pressure"].loc[0, :]
wntr.graphics.plot_network(
    wn,
    node_attribute=pressure_at_5hr,
    node_size=50,
    title="Pressure at 5 hours",
    node_labels=False,
)
