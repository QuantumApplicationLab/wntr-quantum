import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from quantum_newton_raphson.qubo_solver import QUBO_SOLVER

epanet_path = os.environ["EPANET_QUANTUM"]
epanet_tmp = os.environ["EPANET_TMP"]
util_path = os.path.join(epanet_path, "src/py/")
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append(util_path)
from quantum_linsolve import load_json_data

A, b = load_json_data(os.path.join(epanet_tmp, "smat.json"))

linear_solver = QUBO_SOLVER(
    num_qbits=11,
    num_reads=100,
    iterations=5,
    range=1000,
    offset=0,
    temperature=1e4,
    use_aequbols=True,
)

qubo_sol = linear_solver(A.todense(), b)

np_sol = np.linalg.solve(A.todense(), b)


plt.scatter(np_sol, qubo_sol.solution)
plt.axline((0, 0), slope=1, linestyle="--", color="gray")
plt.show()
