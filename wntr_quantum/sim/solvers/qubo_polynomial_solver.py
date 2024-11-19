from collections import OrderedDict
from typing import Dict
from typing import List
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import sparse
from dimod import SampleSet
from dimod import Vartype
from dimod import Sampler
from quantum_newton_raphson.newton_raphson import newton_raphson
from qubops.encodings import BaseQbitEncoding
from qubops.encodings import PositiveQbitEncoding
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
from ...sampler.simulated_annealing import SimulatedAnnealing
from ...sampler.step.full_random import IncrementalStep
from ..models.chezy_manning import get_chezy_manning_qubops_matrix
from ..models.darcy_weisbach import get_darcy_weisbach_qubops_matrix
from ..models.mass_balance import get_mass_balance_qubops_matrix


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
            flow_encoding (qubops.encodings.BaseQbitEncoding): binary encoding for the bsolute value of the flow
            head_encoding (qubops.encodings.BaseQbitEncoding): binary encoding for the head
        """
        self.wn = wn

        # create the encoding vectors for the sign of the flows
        self.sign_flow_encoding = PositiveQbitEncoding(
            nqbit=1, step=2, offset=-1, var_base_name="x"
        )

        # store the encoding of the flow
        self.flow_encoding = flow_encoding
        if np.min(self.flow_encoding.get_possible_values()) < 0:
            raise ValueError(
                "The encoding of the flows must only take positive values."
            )

        # store the encoding of the head
        self.head_encoding = head_encoding

        # create the solution vectors
        self.sol_vect_signs = SolutionVector(
            wn.num_pipes, encoding=self.sign_flow_encoding
        )
        self.sol_vect_flows = SolutionVector(wn.num_pipes, encoding=flow_encoding)
        self.sol_vect_heads = SolutionVector(wn.num_junctions, encoding=head_encoding)

        # create the mixed solution vector
        self.mixed_solution_vector = MixedSolutionVector(
            [self.sol_vect_signs, self.sol_vect_flows, self.sol_vect_heads]
        )

        # init other attributes
        self.matrices = None
        self.qubo = None
        self.flow_index_mapping = None
        self.head_index_mapping = None

        # set up the sampler
        self.sampler = SimulatedAnnealing()

    def verify_encoding(self):
        """Print info regarding the encodings."""
        hres = self.head_encoding.get_average_precision()
        hvalues = np.sort(self.head_encoding.get_possible_values())
        fres = self.flow_encoding.get_average_precision()
        fvalues = np.sort(self.flow_encoding.get_possible_values())
        print("Head Encoding : %f => %f (res: %f)" % (hvalues[0], hvalues[-1], hres))
        print(
            "Flow Encoding : %f => %f | %f => %f (res: %f)"
            % (-fvalues[-1], -fvalues[0], fvalues[0], fvalues[-1], fres)
        )

    def verify_solution(self, input: np.ndarray) -> np.ndarray:
        """Computes the rhs vector associate with the input.

        Args:
            input (np.ndarray): proposed solution

        Returns:
            np.ndarray: RHS vector
        """
        P0, P1, P2, P3 = self.matrices
        num_pipes = self.wn.num_pipes

        if self.wn.options.hydraulic.headloss == "C-M":
            p0 = P0.reshape(
                -1,
            )
            p1 = P1[:, num_pipes:] + P2.sum(1)[:, num_pipes:]
            p2 = P3.sum(1)[:, num_pipes:, num_pipes:].sum(-1)
        elif self.wn.options.hydraulic.headloss == "D-W":
            raise NotImplementedError("verify_solution not implemented for DW")
        sign = np.sign(input)
        return p0 + p1 @ input + (p2 @ (sign * input * input))

    def classical_solution(
        self, model=None, max_iter: int = 100, tol: float = 1e-10
    ) -> np.ndarray:
        """Computes the solution using a classical Newton Raphson approach.

        Args:
            model (model): the model
            max_iter (int, optional): number of iterations of the NR. Defaults to 100.
            tol (float, optional): Toleracne of the NR. Defaults to 1e-10.

        Returns:
            np.ndarray: _description_
        """
        if self.matrices is None:
            self.create_index_mapping(model)
            self.matrices = self.initialize_matrices(model)

        P0, P1, P2, P3 = self.matrices
        num_heads = self.wn.num_junctions
        num_pipes = self.wn.num_pipes
        num_signs = self.wn.num_pipes
        num_vars = num_heads + num_pipes

        if self.wn.options.hydraulic.headloss == "C-M":
            p0 = P0.reshape(
                -1,
            )
            p1 = P1[:, num_pipes:] + P2.sum(1)[:, num_pipes:]
            p2 = P3.sum(1)[:, num_pipes:, num_pipes:].sum(-1)

        elif self.wn.options.hydraulic.headloss == "D-W":
            p0 = P0.reshape(
                -1,
            ) + P1[
                :, :num_signs
            ].sum(-1)
            p1 = P1[:, num_pipes:] + P2.sum(1)[:, num_pipes:]
            p2 = P3.sum(1)[:, num_pipes:, num_pipes:].sum(-1)

        def func(input):
            sign = np.sign(input)
            return p0 + p1 @ input + (p2 @ (sign * input * input))

        initial_point = np.random.rand(num_vars)
        res = newton_raphson(func, initial_point, max_iter=max_iter, tol=tol)
        sol = res.solution
        converged = np.allclose(func(sol), 0)

        # get the closest encoded solution and binary encoding
        bin_rep_sol = []
        for i in range(num_pipes):
            bin_rep_sol.append(int(sol[i] > 0))

        encoded_sol = np.zeros_like(sol)
        for idx, s in enumerate(sol):
            val, bin_rpr = self.mixed_solution_vector.encoded_reals[
                idx + num_pipes
            ].find_closest(np.abs(s))
            bin_rep_sol.append(bin_rpr)
            encoded_sol[idx] = np.sign(s) * val

        # convert back to SI
        sol = self.convert_solution_to_si(sol)
        encoded_sol = self.convert_solution_to_si(encoded_sol)

        # remove the height of the junctions
        for i in range(self.wn.num_junctions):
            sol[num_pipes + i] -= self.wn.nodes[self.wn.junction_name_list[i]].elevation
            encoded_sol[num_pipes + i] -= self.wn.nodes[
                self.wn.junction_name_list[i]
            ].elevation

        return (sol, encoded_sol, bin_rep_sol, converged)

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

    def decompose_solution(self, solution):
        """Decompose solution into sign/abs flow and head values.

        Args:
            solution (np.array): solution
        """
        num_flows = self.wn.num_links
        flow_values = solution[:num_flows]
        head_values = solution[num_flows:]
        tmp = np.append(np.sign(flow_values), np.abs(flow_values))
        return np.append(tmp, head_values)

    def diagnostic_solution(self, solution: np.ndarray, reference_solution: np.ndarray):
        """Benchmark a solution against the exact reference solution.

        Args:
            solution (np.array): solution to be benchmarked
            reference_solution (np.array): reference solution
        """
        reference_solution = self.convert_solution_from_si(reference_solution)
        solution = self.convert_solution_from_si(solution)

        reference_solution = self.decompose_solution(reference_solution)
        solution = self.decompose_solution(solution)

        data_ref, eref = self.qubo.compute_energy(reference_solution)
        data_sol, esol = self.qubo.compute_energy(solution)

        num_pipes = self.wn.num_links

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
        print("E ref   : ", eref)
        print("Delta E :", esol - eref)
        print("\n")
        res_sol = np.linalg.norm(
            self.verify_solution(np.array(data_sol[0][num_pipes:]))
        )
        res_ref = np.linalg.norm(
            self.verify_solution(np.array(data_ref[0][num_pipes:]))
        )
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
        num_variables = 2 * len(model.flow) + len(model.head)

        # must transform that to coo
        P0 = np.zeros((num_equations, 1))
        P1 = np.zeros((num_equations, num_variables))
        P2 = np.zeros((num_equations, num_variables, num_variables))
        P3 = np.zeros((num_equations, num_variables, num_variables, num_variables))
        matrices = (P0, P1, P2, P3)

        # get the mass balance
        matrices = get_mass_balance_qubops_matrix(
            model, self.wn, matrices, self.flow_index_mapping, convert_to_us_unit=True
        )

        # get the headloss matrix contributions
        if self.wn.options.hydraulic.headloss == "C-M":
            matrices = get_chezy_manning_qubops_matrix(
                model,
                self.wn,
                matrices,
                self.flow_index_mapping,
                self.head_index_mapping,
            )
        elif self.wn.options.hydraulic.headloss == "D-W":
            matrices = get_darcy_weisbach_qubops_matrix(
                model,
                self.wn,
                matrices,
                self.flow_index_mapping,
                self.head_index_mapping,
            )
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
    def combine_flow_values(solution: List) -> List:
        """Combine the values of the flow sign*abs.

        Args:
            solution (List): solution vector

        Returns:
            List: solution vector
        """
        flow = []
        for sign, abs in zip(solution[0], solution[1]):
            flow.append(sign * abs)
        return flow + solution[2]

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

    def load_data_in_model(self, model: Model, data: np.ndarray):
        """Loads some data in the model.

        Remark:
            This routine replaces `load_var_values_from_x` without reordering the vector elements

        Args:
            model (Model): AML model from WNTR
            data (np.ndarray): data to load
        """
        shift_head_idx = self.wn.num_links
        for var in model.vars():
            if var.name in self.flow_index_mapping:
                idx = self.flow_index_mapping[var.name]["sign"]
            elif var.name in self.head_index_mapping:
                idx = self.head_index_mapping[var.name] - shift_head_idx
            var.value = data[idx]

    def extract_data_from_model(self, model: Model) -> np.ndarray:
        """Loads some data in the model.

        Args:
            model (Model): AML model from WNTR

        Returns:
            np.ndarray: data extracted from model
        """
        data = [None] * len(list(model.vars()))
        shift_head_idx = self.wn.num_links
        for var in model.vars():
            if var.name in self.flow_index_mapping:
                idx = self.flow_index_mapping[var.name]["sign"]
            elif var.name in self.head_index_mapping:
                idx = self.head_index_mapping[var.name] - shift_head_idx
            data[idx] = var.value
        return data

    def create_index_mapping(self, model: Model) -> None:
        """Creates the index maping for qubops matrices.

        Args:
            model (Model): the AML Model
        """
        # init the idx
        idx = 0

        # number of variables that are flows
        num_flow_var = len(model.flow)

        # get the indices for the sign/abs value of the flow
        self.flow_index_mapping = OrderedDict()
        for _, val in model.flow.items():
            if val.name not in self.flow_index_mapping:
                self.flow_index_mapping[val.name] = {
                    "sign": None,
                    "absolute_value": None,
                }
            self.flow_index_mapping[val.name]["sign"] = idx
            self.flow_index_mapping[val.name]["absolute_value"] = idx + num_flow_var
            idx += 1

        # get the indices for the heads
        idx = 0
        self.head_index_mapping = OrderedDict()
        for _, val in model.head.items():
            self.head_index_mapping[val.name] = 2 * num_flow_var + idx
            idx += 1

    def solve(  # noqa: D417
        self,
        model: Model,
        strength: float = 1e6,
        **sampler_options,
    ) -> Tuple:
        """Solves the Hydraulics equations.

        Args:
            model (Model): AML model
            strength (float, optional): substitution strength. Defaults to 1e6.
            num_reads (int, optional): number of reads for the sampler. Defaults to 10000.

        Returns:
            Tuple: Succes message
        """
        # creates the index mapping for the variables in  the solution vectors
        self.create_index_mapping(model)

        # creates the matrices
        self.matrices = self.initialize_matrices(model)

        # solve using qubo poly
        sol = self.qubo_poly_solve(
            strength=strength, sampler=self.sampler, **sampler_options
        )

        # load data in the AML model
        model.set_structure()
        self.load_data_in_model(model, sol)

        # returns
        return (
            SolverStatus.converged,
            "Solved Successfully",
            0,
        )

    def qubo_poly_solve(
        self,
        strength=1e7,
        **sampler_options,
    ):  # noqa: D417
        """Solves the Hydraulics equations.

        Args:
            strength (float, optional): substitution strength. Defaults to 1e6.
            sampler (float, dwave.sampler):  sampler to optimize the qubo
            **sampler_options (dict): options for the sampler

        Returns:
            np.ndarray: solution of the problem
        """
        self.qubo = QUBOPS_MIXED(self.mixed_solution_vector, {"sampler": self.sampler})
        matrices = tuple(sparse.COO(m) for m in self.matrices)

        # creates BQM
        self.qubo.qubo_dict = self.qubo.create_bqm(matrices, strength=strength)

        # sample
        self.sampleset = self.qubo.sample_bqm(self.qubo.qubo_dict, **sampler_options)

        # decode
        sol = self.qubo.decode_solution(self.sampleset.lowest().record[0][0])

        # combine the sign*abs values for the flow
        sol = self.combine_flow_values(sol)

        # convert back to SI
        sol = self.convert_solution_to_si(sol)

        # remove the height of the junction
        for i in range(self.wn.num_junctions):
            sol[self.wn.num_pipes + i] -= self.wn.nodes[
                self.wn.junction_name_list[i]
            ].elevation

        return sol

    def analyze_sampleset(self):
        """Ananlyze the results contained in the sampleset."""

        # run through all samples
        solutions, energy, quadra_status = [], [], []
        for x in self.sampleset.data():

            # create a sample
            y = SampleSet.from_samples(x.sample, Vartype.BINARY, x.energy)
            var = y.variables
            data = np.array(y.record[0][0])

            # see if it respects quadratic condition
            status = "True"
            for v, d in zip(var, data):
                if v not in self.qubo.mapped_variables:
                    var_tmp = v.split("*")
                    itmp = 0
                    for vtmp in var_tmp:
                        idx = self.qubo.index_variables[
                            self.qubo.mapped_variables.index(vtmp)
                        ]
                        if itmp == 0:
                            dcomposite = data[idx]
                            itmp = 1
                        else:
                            dcomposite *= data[idx]
                    if d != dcomposite:
                        status = False
                        break
            quadra_status.append(status)

            # solution
            sol = self.qubo.decode_solution(data)

            # combine the sign*abs values for the flow
            sol = self.combine_flow_values(sol)

            # convert back to SI
            sol = self.convert_solution_to_si(sol)

            # remove the height of the junction
            for i in range(self.wn.num_junctions):
                sol[self.wn.num_pipes + i] -= self.wn.nodes[
                    self.wn.junction_name_list[i]
                ].elevation

            solutions.append(sol)
            energy.append(x.energy)
        return solutions, energy, quadra_status
