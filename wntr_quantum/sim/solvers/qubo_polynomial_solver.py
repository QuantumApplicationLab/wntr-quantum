import matplotlib.pyplot as plt
import numpy as np
import sparse
from quantum_newton_raphson.newton_raphson import newton_raphson
from qubols.encodings import BaseQbitEncoding
from qubols.encodings import DiscreteValuesEncoding
from qubols.mixed_solution_vector import MixedSolutionVector_V2 as MixedSolutionVector
from qubols.qubo_poly_mixed_variables import QUBO_POLY_MIXED
from qubols.solution_vector import SolutionVector_V2 as SolutionVector
from wntr.epanet.util import FlowUnits
from wntr.epanet.util import HydParam
from wntr.epanet.util import from_si
from wntr.epanet.util import to_si
from wntr.sim.solvers import SolverStatus
from ..models.chezy_manning import get_chezy_manning_matrix
from ..models.darcy_weisbach import get_darcy_weisbach_matrix
from ..models.mass_balance import get_mass_balance_matrix


class QuboPolynomialSolver(object):
    """Solve the hydraulics equation following a QUBO approach."""

    def __init__(
        self,
        wn,
        flow_encoding,
        head_encoding,
    ):  # noqa: D417
        """Init the solver.

        Args:
            wn (WaterNetwork): water network
            flow_encoding (BaseEncoding): binary encoding for the flow
            head_encoding (BaseEncoding): binary encoding for the head            pipe_diameters (_type_): _description_
        """
        self.wn = wn

        # create the encoding vectors
        self.flow_encoding = flow_encoding
        self.head_encoding = head_encoding
        self.sol_vect_flows = SolutionVector(wn.num_pipes, encoding=flow_encoding)
        self.sol_vect_heads = SolutionVector(wn.num_junctions, encoding=head_encoding)
        self.mixed_solution_vector = MixedSolutionVector(
            [self.sol_vect_flows, self.sol_vect_heads]
        )

    def verify_encoding(self):
        """Print info regarding the encodings."""
        hres = self.head_encoding.get_average_precision()
        hvalues = np.sort(self.head_encoding.get_possible_values())
        fres = self.flow_encoding.get_average_precision()
        fvalues = np.sort(self.flow_encoding.get_possible_values())
        print("Head Encoding : %f => %f (res: %f)" % (hvalues[0], hvalues[-1], hres))
        print("Flow Encoding : %f => %f (res: %f)" % (fvalues[0], fvalues[-1], fres))

    def verify_solution(self, input):
        """Generates the classical solution."""
        P0, P1, P2 = self.matrices

        p0 = P0.reshape(
            -1,
        )
        p1 = P1
        p2 = P2.sum(-1)
        return p0 + p1 @ input + (p2 @ (input * input))

    def classical_solutions(self, max_iter=100, tol=1e-10):
        """Computes the classical solution."""
        P0, P1, P2 = self.matrices
        num_heads = self.wn.num_junctions
        num_pipes = self.wn.num_pipes
        num_vars = num_heads + num_pipes

        p0 = P0.reshape(
            -1,
        )
        p1 = P1
        p2 = P2.sum(-1)

        def func(input):
            return p0 + p1 @ input + (p2 @ (input * input))

        initial_point = np.random.rand(num_vars)
        res = newton_raphson(func, initial_point, max_iter=max_iter, tol=tol)
        sol = res.solution
        assert np.allclose(func(sol), 0)

        # convert back to SI if DW
        sol = self.convert_solution_to_si(sol)

        return sol

    @staticmethod
    def plot_solution_vs_reference(solution, reference_solution):
        """Plots the scatter plot ref/sol.

        Args:
            solution (_type_): _description_
            reference_solution (_type_): _description_
        """
        plt.scatter(reference_solution, solution)
        plt.axline((0, 0.0), slope=1, color="black", linestyle=(0, (5, 5)))

        plt.axline((0, 0.0), slope=1.05, color="grey", linestyle=(0, (2, 2)))
        plt.axline((0, 0.0), slope=0.95, color="grey", linestyle=(0, (2, 2)))
        plt.grid(which="major", lw=1)
        plt.grid(which="minor", lw=0.1)
        plt.loglog()

    def benchmark_solution(self, solution, reference_solution, qubo, bqm):
        """Benchmark a solution against the exact reference solution.

        Args:
            solution (np.array): _description_
            reference_solution (np.array): _description_
            qubo (_type_): __
            bqm (_type_): __
        """
        reference_solution = self.convert_solution_from_si(reference_solution)
        solution = self.convert_solution_from_si(solution)

        data_ref, eref = qubo.compute_energy(reference_solution, bqm)
        data_sol, esol = qubo.compute_energy(solution, bqm)

        np.set_printoptions(precision=3)
        self.verify_encoding()
        print("\n")
        print("Error (%):", (1 - (solution / reference_solution)) * 100)
        print("\n")
        print("sol : ", solution)
        print("ref : ", reference_solution)
        print("diff: ", reference_solution - solution)
        print("\n")
        print("encoded_sol: ", np.array(data_sol[0]))
        print("encoded_ref: ", np.array(data_ref[0]))
        print("diff       : ", np.array(data_ref[0]) - np.array(data_sol[0]))
        print("\n")
        print("E sol   : ", esol)
        print("R ref   : ", eref)
        print("Delta E :", esol - eref)
        print("\n")
        res_sol = np.linalg.norm(self.verify_solution(np.array(data_sol[0])))
        res_ref = np.linalg.norm(self.verify_solution(np.array(data_ref[0])))
        print("Residue sol   : ", res_sol)
        print("Residue ref   : ", res_ref)
        print("Delta Residue :", res_sol - res_ref)

    def initialize_matrices(self, model):
        """Initilize the matrix for the QUBO definition."""
        num_equations = len(list(model.cons()))
        num_variables = len(list(model.vars()))

        # must transform that to coo
        P0 = np.zeros((num_equations, 1))
        P1 = np.zeros((num_equations, num_variables))
        P2 = np.zeros((num_equations, num_variables, num_variables))

        matrices = (P0, P1, P2)

        # get the mass balance
        matrices = get_mass_balance_matrix(
            model, self.wn, matrices, convert_to_us_unit=True
        )

        # get the headloss matrix contributions
        if self.wn.options.hydraulic.headloss == "C-M":
            matrices = get_chezy_manning_matrix(model, self.wn, matrices)
        elif self.wn.options.hydraulic.headloss == "D-W":
            matrices = get_darcy_weisbach_matrix(model, self.wn, matrices)
        else:
            raise ValueError("Calculation only possible with C-M or D-W")
        return matrices

    def convert_solution_to_si(self, solution):
        """Converts the solution to SI.

        Args:
            solution (array): solution vectors
        """
        num_heads = self.wn.num_junctions
        num_pipes = self.wn.num_pipes
        new_sol = np.zeros_like(solution)
        for ip in range(num_pipes):
            new_sol[ip] = to_si(FlowUnits.CFS, solution[ip], HydParam.Flow)
        for ih in range(num_pipes, num_pipes + num_heads):
            new_sol[ih] = to_si(FlowUnits.CFS, solution[ih], HydParam.Length)
        return new_sol

    @staticmethod
    def flatten_solution_vector(solution):
        """Flattens the solution vector.

        Args:
            solution (tuple): tuple of ([flows], [heads])
        """
        sol_tmp = []
        for s in solution:
            sol_tmp += s
        return sol_tmp

    def convert_solution_from_si(self, solution):
        """Converts the solution to SI.

        Args:
            solution (array): solution vectors
        """
        num_heads = self.wn.num_junctions
        num_pipes = self.wn.num_pipes
        new_sol = np.zeros_like(solution)
        for ip in range(num_pipes):
            new_sol[ip] = from_si(FlowUnits.CFS, solution[ip], HydParam.Flow)
        for ih in range(num_pipes, num_pipes + num_heads):
            new_sol[ih] = from_si(FlowUnits.CFS, solution[ih], HydParam.Length)
        return new_sol

    @staticmethod
    def load_data_in_model(model, data):
        """Loads some data in the model.

        Args:
            model (_type_): _description_
            data (_type_): _description_
        """
        for iv, v in enumerate(model.vars()):
            v.value = data[iv]

    def solve(self, model, strength=1e6, num_reads=10000, **options):
        """Solve the hydraulics equations."""
        self.matrices = self.initialize_matrices(model)
        sol = self.solve_(strength=strength, num_reads=num_reads, **options)
        model.set_structure()
        self.load_data_in_model(model, sol)
        return (
            SolverStatus.converged,
            "Solved Successfully",
            0,
        )

    def solve_(self, strength=1e6, num_reads=10000, **options):
        """Solve the hydraulic equations."""
        qubo = QUBO_POLY_MIXED(self.mixed_solution_vector, **options)
        matrices = tuple(sparse.COO(m) for m in self.matrices)
        bqm = qubo.create_bqm(matrices, strength=strength)

        # sample
        sampleset = qubo.sample_bqm(bqm, num_reads=num_reads)

        # decode
        sol = qubo.decode_solution(sampleset.lowest().record[0][0])

        # flatten solution
        sol = self.flatten_solution_vector(sol)

        # convert back to SI if DW
        sol = self.convert_solution_to_si(sol)

        return sol
