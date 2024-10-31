import wntr
import wntr_quantum
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from wntr_quantum.sim.solvers.qubo_polynomial_solver import QuboPolynomialSolver
from qubops.solution_vector import SolutionVector_V2 as SolutionVector
from qubops.encodings import RangedEfficientEncoding, PositiveQbitEncoding
from wntr_quantum.sim.qubo_hydraulics import create_hydraulic_model_for_qubo
from wntr_quantum.sampler.simulated_annealing import SimulatedAnnealing
from qubops.qubops_mixed_vars import QUBOPS_MIXED
import sparse
from wntr_quantum.sampler.step.full_random import IncrementalStep
from wntr_quantum.sampler.simulated_annealing import modify_solution_sample

import pickle


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
inp_file = "./networks/Net0.inp"
# inp_file = './networks/Net2LoopsDW.inp'
wn_ref = wntr.network.WaterNetworkModel(inp_file)

# store the results
qubo_results = []
solutions = []
encoded_reference_solutions = []

# iterate over a bunch of confs
Nsim = 100
for i in range(Nsim):
    print("==== %d / %d ====" % (i, Nsim))
    # copy the nework
    wn = deepcopy(wn_ref)

    # change pipe diams
    # for pipe_name in wn.link_name_list:
    #     pipe = wn.get_link(pipe_name)
    #     eps = 0.9 + 0.2 * np.random.rand()
    #     pipe.diameter *= eps

    # solve classcaly
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    # extract ref values
    ref_pressure = results.node["pressure"].values[0][:2]
    ref_rate = results.link["flowrate"].values[0]
    ref_values = np.append(ref_rate, ref_pressure)
    ref_values

    # create qubo encoding for the flow
    nqbit = 7
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

    # create qubosolver
    net = QuboPolynomialSolver(
        wn, flow_encoding=flow_encoding, head_encoding=head_encoding
    )

    # create model
    model, model_updater = create_hydraulic_model_for_qubo(wn)
    net.create_index_mapping(model)
    net.matrices = net.initialize_matrices(model)

    # solve qubo classically
    ref_sol, encoded_ref_sol, bin_rep_sol, cvgd = net.classical_solutions()
    encoded_reference_solutions.append(encoded_ref_sol)

    # sampler
    sampler = SimulatedAnnealing()

    # create the solver attribute
    net.qubo = QUBOPS_MIXED(net.mixed_solution_vector, {"sampler": sampler})
    matrices = tuple(sparse.COO(m) for m in net.matrices)
    net.qubo.qubo_dict = net.qubo.create_bqm(matrices, strength=0)

    # create step
    var_names = sorted(net.qubo.qubo_dict.variables)
    net.qubo.create_variables_mapping()
    mystep = IncrementalStep(
        var_names, net.qubo.mapped_variables, net.qubo.index_variables, step_size=1
    )

    # generate init sample
    # x = modify_solution_sample(net, bin_rep_sol, modify=["flows", "heads"])
    x = modify_solution_sample(net, bin_rep_sol, modify=["heads"])
    x0 = list(x.values())

    # compute ref energy
    eref = net.qubo.energy_binary_rep(bin_rep_sol)

    # temperature schedule
    num_sweeps = 2000
    Tinit = 1e1
    Tfinal = 1e-1
    Tschedule = np.linspace(Tinit, Tfinal, num_sweeps)
    Tschedule = np.append(Tschedule, Tfinal * np.ones(1000))
    Tschedule = np.append(Tschedule, np.zeros(1000))

    # sample
    mystep.optimize_values = np.arange(4, 6)
    res = sampler.sample(
        net.qubo,
        init_sample=x0,
        Tschedule=Tschedule,
        take_step=mystep,
        save_traj=True,
        verbose=False,
    )
    mystep.verify_quadratic_constraints(res.res)
    qubo_results.append(res)

    # compute final
    idx_min = np.array([e for e in res.energies]).argmin()
    # idx_min = -1
    sol = res.trajectory[idx_min]
    sol = net.qubo.decode_solution(np.array(sol))
    sol = net.combine_flow_values(sol)
    sol = net.convert_solution_to_si(sol)
    solutions.append(sol)


plot_solutions(solutions, encoded_reference_solutions)
pickle.dump(solutions, open("solutions.pkl", "wb"))
pickle.dump(encoded_reference_solutions, open("encoded_reference_solutions.pkl", "wb"))
# pickle.dump(qubo_results, open("qubo_results.pkl", "wb"))
