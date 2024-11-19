import pickle
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import sparse
import wntr
from qubops.encodings import PositiveQbitEncoding
from qubops.qubops_mixed_vars import QUBOPS_MIXED
from wntr_quantum.design.qubo_pipe_diam import QUBODesignPipeDiameter
from wntr_quantum.sampler.simulated_annealing import SimulatedAnnealing
from wntr_quantum.sampler.simulated_annealing import modify_solution_sample
from wntr_quantum.sampler.step.full_random import SwitchIncrementalStep


def plot_solutions(solutions, references):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(121)

    ax1.axline((0, 0.0), slope=1.10, color="grey", linestyle=(0, (2, 5)))
    ax1.axline((0, 0.0), slope=1, color="black", linestyle=(0, (2, 5)))
    ax1.axline((0, 0.0), slope=0.90, color="grey", linestyle=(0, (2, 5)))
    ax1.grid()

    for r, sol in zip(references, solutions):
        ax1.scatter(
            r[:2], sol[:2], s=150, lw=1, edgecolors="w", label="Sampled solution"
        )

    ax1.set_xlabel("Reference Values", fontsize=12)
    ax1.set_ylabel("QUBO Values", fontsize=12)
    ax1.set_title("Flow Rate", fontsize=14)

    ax2 = fig.add_subplot(122)

    ax2.axline((0, 0.0), slope=1.10, color="grey", linestyle=(0, (2, 5)))
    ax2.axline((0, 0.0), slope=1, color="black", linestyle=(0, (2, 5)))
    ax2.axline((0, 0.0), slope=0.90, color="grey", linestyle=(0, (2, 5)))

    for r, sol in zip(references, solutions):
        ax2.scatter(
            r[2:],
            sol[2:],
            s=150,
            lw=1,
            edgecolors="w",
            label="Sampled solution",
        )
    ax2.grid()

    ax2.set_xlabel("Reference Values", fontsize=12)
    ax2.set_title("Pressure", fontsize=14)
    plt.show()


# Create a water network model
inp_file = "./networks/Net0_CM.inp"
# inp_file = "./networks/Net0.inp"
# inp_file = './networks/Net2LoopsDW.inp'
wn_ref = wntr.network.WaterNetworkModel(inp_file)

# store the results
energies = []
solutions = []
encoded_reference_solutions = []
prices = []
optimal_diameters = []

# iterate over a bunch of confs
Nsim = 100
for i in range(Nsim):
    print("==== %d / %d ====" % (i, Nsim))
    # copy the nework
    wn = deepcopy(wn_ref)

    # solve classcaly
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    # extract ref values
    ref_pressure = results.node["pressure"].values[0][:2]
    ref_rate = results.link["flowrate"].values[0]
    ref_values = np.append(ref_rate, ref_pressure)
    ref_values

    # create qubo encoding for the flow
    nqbit = 5
    step = 4.0 / (2**nqbit - 1)
    flow_encoding = PositiveQbitEncoding(
        nqbit=nqbit, step=step, offset=+0, var_base_name="x"
    )

    # create qubo encoding for the heads
    nqbit = 7
    step = 200 / (2**nqbit - 1)
    head_encoding = PositiveQbitEncoding(
        nqbit=nqbit, step=step, offset=+0.0, var_base_name="x"
    )

    # create designer
    pipe_diameters = [250, 500, 1000]
    designer = QUBODesignPipeDiameter(
        wn,
        flow_encoding,
        head_encoding,
        pipe_diameters,
        head_lower_bound=95,
        weight_cost=2,
        weight_pressure=0.5,
    )

    # create model
    designer.create_index_mapping()
    designer.matrices = designer.initialize_matrices()
    ref_sol, encoded_ref_sol, bin_rep_sol, cvgd = designer.classical_solution(
        [0, 1, 0, 0, 1, 0], convert_to_si=True
    )

    # sampler
    sampler = SimulatedAnnealing()

    # create the solver attribute
    designer.qubo = QUBOPS_MIXED(designer.mixed_solution_vector, {"sampler": sampler})
    matrices = tuple(sparse.COO(m) for m in designer.matrices)
    designer.qubo.qubo_dict = designer.qubo.create_bqm(matrices, strength=0)
    # designer.add_switch_constraints(strength=0)
    designer.add_pressure_equality_constraints()

    # create step
    var_names = sorted(designer.qubo.qubo_dict.variables)
    designer.qubo.create_variables_mapping()
    mystep = SwitchIncrementalStep(
        var_names,
        designer.qubo.mapped_variables,
        designer.qubo.index_variables,
        step_size=10,
        switch_variable_index=[[6, 7, 8], [9, 10, 11]],
    )

    # generate init sample
    # x = modify_solution_sample(net, bin_rep_sol, modify=["flows", "heads"])
    x = modify_solution_sample(designer, bin_rep_sol, modify=["flows", "heads"])
    x0 = list(x.values())

    # temperature schedule
    num_sweeps = 5000
    Tinit = 1e3
    Tfinal = 1e-1
    Tschedule = np.linspace(Tinit, Tfinal, num_sweeps)
    Tschedule = np.append(Tschedule, Tfinal * np.ones(1000))
    Tschedule = np.append(Tschedule, np.zeros(100))

    # sample flow
    mystep.optimize_values = np.arange(2, 12)
    res = sampler.sample(
        designer.qubo,
        init_sample=x0,
        Tschedule=Tschedule,
        take_step=mystep,
        save_traj=True,
        verbose=False,
    )
    mystep.verify_quadratic_constraints(res.res)

    idx_min = np.array([e for e in res.energies]).argmin()
    energies.append(res.energies[idx_min])
    # idx_min = -1
    sol = res.trajectory[idx_min]
    sol = designer.qubo.decode_solution(np.array(sol))
    pipe_hot_encoding = sol[3]
    sol = designer.combine_flow_values(sol)
    sol = designer.convert_solution_to_si(sol)
    sol = sol[:4]
    solutions.append(sol)

    price, diameters = designer.get_pipe_info_from_hot_encoding(pipe_hot_encoding)
    prices.append(price)
    optimal_diameters.append(diameters)

data = {}
for opt, e in zip(optimal_diameters, energies):
    if tuple(opt) not in data:
        data[tuple(opt)] = []
    data[tuple(opt)].append(e[0])

vals = []
labels = []
for k, v in data.items():
    labels.append(k)
    vals.append(v)

width = np.array([(np.array(optimal_diameters) == l).prod(1).sum() for l in labels])
width = 0.5 * width / np.max(width)


plt.violinplot(vals, widths=width)
plt.xticks(list(range(1, 1 + len(labels))), labels)
plt.grid()
plt.show()

# plot_solutions(solutions, encoded_reference_solutions)
pickle.dump(prices, open("prices.pkl", "wb"))
pickle.dump(optimal_diameters, open("optimized_diameters.pkl", "wb"))
pickle.dump(energies, open("energies.pkl", "wb"))
# pickle.dump(qubo_results, open("qubo_results.pkl", "wb"))
