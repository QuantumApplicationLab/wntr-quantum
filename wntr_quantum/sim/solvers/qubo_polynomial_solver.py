from typing import List
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import sparse
from quantum_newton_raphson.newton_raphson import newton_raphson
from qubops.encodings import BaseQbitEncoding
from qubops.mixed_solution_vector import MixedSolutionVector_V2 as MixedSolutionVector
from qubops.qubops_mixed_vars import QUBOPS_MIXED
from qubops.solution_vector import SolutionVector_V2 as SolutionVector
from wntr.epanet.util import FlowUnits
from wntr.epanet.util import HydParam
from wntr.epanet.util import from_si
from wntr.epanet.util import to_si
from wntr.network import WaterNetworkModel
from wntr.sim.aml import Model
from wntr.sim.solvers import SolverStatus
from ..models.chezy_manning import get_chezy_manning_matrix
from ..models.darcy_weisbach import get_darcy_weisbach_matrix
from ..models.mass_balance import get_mass_balance_matrix


class QuboPolynomialSolver(object):
    """Solve the hydraulics equation following a QUBO approach."""

    def __init__(
        self,
        wn: WaterNetworkModel,
        flow_encoding: BaseQbitEncoding,
        head_encoding: BaseQbitEncoding,
    ):  # noqa: D417
        """Init the solver.

        Args:
            wn (WaterNetworkModel): water network
            flow_encoding (qubops.encodings.BaseQbitEncoding): binary encoding for the flow
            head_encoding (qubops.encodings.BaseQbitEncoding): binary encoding for the head
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

        # init other attributes
        self.matrices = None
        self.qubo = None

    def verify_encoding(self):
        """Print info regarding the encodings."""
        hres = self.head_encoding.get_average_precision()
        hvalues = np.sort(self.head_encoding.get_possible_values())
        fres = self.flow_encoding.get_average_precision()
        fvalues = np.sort(self.flow_encoding.get_possible_values())
        print("Head Encoding : %f => %f (res: %f)" % (hvalues[0], hvalues[-1], hres))
        print("Flow Encoding : %f => %f (res: %f)" % (fvalues[0], fvalues[-1], fres))

    def verify_solution(self, input: np.ndarray) -> np.ndarray:
        """Computes the rhs vector associate with the input.

        Args:
            input (np.ndarray): proposed solution

        Returns:
            np.ndarray: RHS vector
        """
        P0, P1, P2 = self.matrices

        p0 = P0.reshape(
            -1,
        )
        p1 = P1
        p2 = P2.sum(-1)
        return p0 + p1 @ input + (p2 @ (input * input))

    def classical_solutions(
        self, max_iter: int = 100, tol: float = 1e-10
    ) -> np.ndarray:
        """Computes the solution using a classical Newton Raphson approach.

        Args:
            max_iter (int, optional): number of iterations of the NR. Defaults to 100.
            tol (float, optional): Toleracne of the NR. Defaults to 1e-10.

        Returns:
            np.ndarray: _description_
        """
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

        # convert back to SI
        sol = self.convert_solution_to_si(sol)

        return sol

    @staticmethod
    def plot_solution_vs_reference(
        solution: np.ndarray, reference_solution: np.ndarray
    ):
        """Plots the scatter plot ref/sol.

        Args:
            solution (np.ndarray): _description_
            reference_solution (np.ndarray): _description_
        """
        plt.scatter(reference_solution, solution)
        plt.axline((0, 0.0), slope=1, color="black", linestyle=(0, (5, 5)))

        plt.axline((0, 0.0), slope=1.05, color="grey", linestyle=(0, (2, 2)))
        plt.axline((0, 0.0), slope=0.95, color="grey", linestyle=(0, (2, 2)))
        plt.grid(which="major", lw=1)
        plt.grid(which="minor", lw=0.1)
        plt.loglog()

    def diagnostic_solution(self, solution: np.ndarray, reference_solution: np.ndarray):
        """Benchmark a solution against the exact reference solution.

        Args:
            solution (np.array): solution to be benchmarked
            reference_solution (np.array): reference solution
        """
        reference_solution = self.convert_solution_from_si(reference_solution)
        solution = self.convert_solution_from_si(solution)

        data_ref, eref = self.qubo.compute_energy(reference_solution)
        data_sol, esol = self.qubo.compute_energy(solution)

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

    def initialize_matrices(self, model: Model) -> Tuple:
        """Initialize the matrices of the non linear system.

        Args:
            model (Model): an AML model from WNTR

        Raises:
            ValueError: if headloss approximation is not C-M or D-W

        Returns:
            Tuple: Matrices of the on linear system
        """
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

    def convert_solution_to_si(self, solution: np.ndarray) -> np.ndarray:
        """Converts the solution to SI.

        Args:
            solution (array): solution vectors in US units

        Returns:
            Tuple: solution in SI
        """
        num_heads = self.wn.num_junctions
        num_pipes = self.wn.num_pipes
        new_sol = np.zeros_like(solution)
        for ip in range(num_pipes):
            new_sol[ip] = to_si(FlowUnits.CFS, solution[ip], HydParam.Flow)
        for ih in range(num_pipes, num_pipes + num_heads):
            new_sol[ih] = to_si(FlowUnits.CFS, solution[ih], HydParam.Length)
        return new_sol

    def convert_solution_from_si(self, solution: np.ndarray) -> np.ndarray:
        """Converts the solution to SI.

        Args:
            solution (array): solution vectors in SI

        Returns:
            Tuple: solution in US units
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
    def flatten_solution_vector(solution: Tuple) -> List:
        """Flattens the solution vector.

        Args:
            solution (tuple): tuple of ([flows], [heads])

        Returns:
            List: a flat list of all the variables
        """
        sol_tmp = []
        for s in solution:
            sol_tmp += s
        return sol_tmp

    @staticmethod
    def load_data_in_model(model: Model, data: np.ndarray):
        """Loads some data in the model.

        Remark:
            This routine replaces `load_var_values_from_x` without reordering the vector elements

        Args:
            model (Model): AML model from WNTR
            data (np.ndarray): data to load
        """
        for iv, v in enumerate(model.vars()):
            v.value = data[iv]

    @staticmethod
    def extract_data_from_model(model: Model) -> np.ndarray:
        """Loads some data in the model.

        Args:
            model (Model): AML model from WNTR

        Returns:
            np.ndarray: data extracted from model
        """
        data = []
        for v in model.vars():
            data.append(v.value)
        return data

    def solve(  # noqa: D417
        self, model: Model, strength: float = 1e6, num_reads: int = 10000, **options
    ) -> Tuple:
        """Solves the Hydraulics equations.

        Args:
            model (Model): AML model
            strength (float, optional): substitution strength. Defaults to 1e6.
            num_reads (int, optional): number of reads for the sampler. Defaults to 10000.

        Returns:
            Tuple: Succes message
        """
        # creates the matrices
        self.matrices = self.initialize_matrices(model)

        # solve using qubo poly
        sol = self.qubo_poly_solve(strength=strength, num_reads=num_reads, **options)

        # load data in the AML model
        model.set_structure()
        self.load_data_in_model(model, sol)

        # returns
        return (
            SolverStatus.converged,
            "Solved Successfully",
            0,
        )

    def qubo_poly_solve(self, strength=1e6, num_reads=10000, **options):  # noqa: D417
        """Solves the Hydraulics equations.

        Args:
            strength (float, optional): substitution strength. Defaults to 1e6.
            num_reads (int, optional): number of reads for the sampler. Defaults to 10000.

        Returns:
            np.ndarray: solution of the problem
        """
        self.qubo = QUBOPS_MIXED(self.mixed_solution_vector, **options)
        matrices = tuple(sparse.COO(m) for m in self.matrices)

        # creates BQM
        self.qubo.qubo_dict = self.qubo.create_bqm(matrices, strength=strength)

        # sample
        sampleset = self.qubo.sample_bqm(self.qubo.qubo_dict, num_reads=num_reads)

        # decode
        sol = self.qubo.decode_solution(sampleset.lowest().record[0][0])

        # flatten solution
        sol = self.flatten_solution_vector(sol)

        # convert back to SI
        sol = self.convert_solution_to_si(sol)

        return sol
