import wntr
import wntr_quantum
import numpy as np
import pickle
import os
from pathlib import Path
import shutil
import time

# for the quantum algorithm
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Estimator
from qiskit_algorithms import optimizers as opt
from quantum_newton_raphson.vqls_solver import VQLS_SOLVER


benchmark = os.getcwd() + '/Pipe_Rough_Coeffs_Scenarios_'

# demand-driven (DD) or pressure dependent demand (PDD)
Mode_Simulation = 'PDD'

# set inp file
INP = "Net2Loops_modified"
print(f"Run input file: {INP}")
inp_file = '../networks/' + INP + '.inp'


def quantum_algo(wn):
    """XXX"""
    qc = RealAmplitudes(num_qubits=3, reps=3, entanglement="full")
    estimator = Estimator()

    linear_solver = VQLS_SOLVER(
        estimator=estimator,
        ansatz=qc,
        optimizer=[opt.COBYLA(maxiter=1000, disp=True), opt.CG(maxiter=500, disp=True)],
        matrix_decomposition="symmetric",
        verbose=False,
        preconditioner="diagonal_scaling",
        reorder=True,
    )

    sim = wntr_quantum.sim.QuantumEpanetSimulator(wn, linear_solver=linear_solver)
    return sim.run_sim(linear_solver=linear_solver)


def runRoughnessCoeffsScenarios(scNum):
    """XXX"""

    itsok = False

    while itsok is not True:

        try:
            # set up interval of roughness coefficients
            min_coeff = 100.0
            max_coeff = 150.0
            interval = 5.0
            roughness_coefficients = np.arange(min_coeff, (max_coeff + interval), interval)
            
            #print(coeffs)

            # path of EPANET input File
            print("Scenarios: " + str(scNum))

            wn = wntr.network.WaterNetworkModel(inp_file)
            inp = os.path.basename(wn.name)[0:-4]

            netName = benchmark + inp

            # Create folder with network name
            if scNum == 1:
                try:
                    if os.path.exists(netName):
                        shutil.rmtree(netName)
                    os.makedirs(netName)
                    shutil.copyfile(inp_file, netName + '/' + os.path.basename(wn.name))
                except:
                    pass

            # Set time parameters
            wn.options.time.duration = 0
            wn.options.hydraulic.demand_model = Mode_Simulation
            wn.options.hydraulic.required_pressure = 30.0  # m
            wn.options.hydraulic.minimum_pressure = 0.0  # m

            wn.options.hydraulic.accuracy = 0.1

            results = {}

            #print(qunc, qunc_index, uncertainty_Length)

            # build scenario folder
            Sc = netName + '/Scenario-' + str(scNum)
            if os.path.exists(Sc):
                shutil.rmtree(Sc)
            os.makedirs(Sc)

            # Roughness coeffs
            tempRoughness = wn.query_link_attribute('roughness')
            tempRoughness = np.array([tempRoughness.iloc[i] for i, _ in enumerate(tempRoughness)])

            sorted_coeffs = np.random.choice(
                roughness_coefficients,
                size=len(tempRoughness),
                replace=False,
            )

            #print(roughness_coefficients, sorted_coeffs)

            for n, coeff in enumerate(sorted_coeffs):
                print(n, wn.get_link(wn.link_name_list[n]).roughness)
                wn.get_link(wn.link_name_list[n]).roughness = coeff
                print(n, wn.get_link(wn.link_name_list[n]).roughness)

            # save EPANET inp file
            epanet_filename = Sc + '/' + inp + '_Scenario-' + str(scNum) + '.inp'
            epanet_io = wntr.epanet.io.InpFile()
            epanet_io.write(epanet_filename, wn)

            # run epanet simulator
            sim = wntr.sim.EpanetSimulator(wn)
            results = sim.run_sim()
            pressures = results.node['pressure']
            flows = results.link['flowrate']
            velocity = results.link["velocity"]

            print(pressures)
            print(flows)
            print(velocity)

            if not ((pressures >= 0) & (pressures <= 151)).all().all():
            #if not ((pressures > 0).all().all()):
                print("not run")
                print(pressures)
                scNum = scNum + 1
                return -1

            # save epanet/wntr `SimulationResults`
            classical_results_pkl = Sc + '/Scenario-' + str(scNum) + '_epanet_results.pkl'
            with open(classical_results_pkl, 'wb') as f:
                pickle.dump(results, f)

            # run quantum algorithm
            try:
                results = {}
                results = quantum_algo(wn)

                pressures = results.node['pressure']
                flows = results.link['flowrate']
                velocity = results.link["velocity"]

                print(pressures)
                print(flows)
                print(velocity)

            except:
                print("Quantum Agorithm failed...")
                scNum = scNum + 1
                return -1

            if results:
                # save results
                flows_file = Sc + '/Scenario-' + str(scNum) + '_flows.csv'
                pressures_file = Sc + '/Scenario-' + str(scNum) + '_pressures.csv'
                flows.to_csv(flows_file, index=False)
                pressures.to_csv(pressures_file, index=False)

                # save `SimulationResults`
                results_pkl = Sc + '/Scenario-' + str(scNum) + '_results.pkl'
                with open(results_pkl, 'wb') as f:
                    pickle.dump(results, f)

                # save `WaterNetworkModel`
                wn_pkl = Sc + '/Scenario-' + str(scNum) + '_wn.pkl'
                with open(wn_pkl, 'wb') as f:
                    pickle.dump(wn, f)

                itsok = True

            else:
                print('results empty')
                return -1

        except:
            itsok = False

    return 1


if __name__ == '__main__':

    t = time.time()

    NumScenarios = 500
    scArray = range(1, NumScenarios)

    for i in list(range(1, NumScenarios+1)):
        runRoughnessCoeffsScenarios(i)

    print('Total Elapsed time is ' + str(time.time() - t) + ' seconds.')
