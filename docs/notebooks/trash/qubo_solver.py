import dimod
import wntr
import wntr_quantum
from quantum_newton_raphson.qubo_solver import QUBO_SOLVER


# Create a water network model
inp_file = "networks/Net0.inp"
wn = wntr.network.WaterNetworkModel(inp_file)

# Graph the network
# wntr.graphics.plot_network(wn, title=wn.name, node_labels=True)

# classical solution
# sim = wntr_quantum.sim.QuantumWNTRSimulator(wn)
# results = sim.run_sim()

# Simulate hydraulics
wn = wntr.network.WaterNetworkModel(inp_file)

# define the linear solver with the reorder solver
linear_solver = QUBO_SOLVER(
    num_qbits=11,
    num_reads=100,
    iterations=5,
    range=150,
    offset=0,
    temperature=1e4,
    use_aequbols=True,
)
sim = wntr_quantum.sim.QuantumWNTRSimulator(wn, linear_solver=linear_solver)
results = sim.run_sim(
    linear_solver=linear_solver, solver_options={"TOL": 1e-2, "FIXED_POINT": False}
)
