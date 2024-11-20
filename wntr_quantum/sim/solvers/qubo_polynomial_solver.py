from collections import OrderedDict
from typing import List
from typing import Tuple
import numpy as np
import sparse
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
from ...sampler.step.random_step import IncrementalStep
from ...sim.qubo_hydraulics import create_hydraulic_model_for_qubo
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
            flow_encoding (qubops.encodings.BaseQbitEncoding): binary encoding for the absolute value of the flow
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

        # create the hydraulics model
        self.model, self.model_updater = create_hydraulic_model_for_qubo(wn)

        # set up the sampler
        self.sampler = SimulatedAnnealing()

        # create the matrices
        self.create_index_mapping(self.model)
        self.matrices = self.initialize_matrices(self.model)
        self.matrices = tuple(sparse.COO(m) for m in self.matrices)

        # create the QUBO MIXED instance
        self.qubo = QUBOPS_MIXED(self.mixed_solution_vector, {"sampler": self.sampler})

        # create the qubo dictionary
        self.qubo.qubo_dict = self.qubo.create_bqm(self.matrices, strength=0)

        self.qubo.create_variables_mapping()
        self.var_names = sorted(self.qubo.qubo_dict.variables)

        # create the step function
        self.step_func = IncrementalStep(
            self.var_names,
            self.qubo.mapped_variables,
            self.qubo.index_variables,
            step_size=10,
        )

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

    def classical_solution(self, max_iter: int = 100, tol: float = 1e-10) -> np.ndarray:
        """Computes the solution using a classical Newton Raphson approach.

        Args:
            model (model): the model
            max_iter (int, optional): number of iterations of the NR. Defaults to 100.
            tol (float, optional): Toleracne of the NR. Defaults to 1e-10.

        Returns:
            np.ndarray: _description_
        """
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

        # compute the qubo energy of the solution
        eref = self.qubo.energy_binary_rep(bin_rep_sol)

        return (sol, encoded_sol, bin_rep_sol, eref, converged)

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
        self, init_sample, Tschedule, save_traj=False, verbose=False
    ) -> Tuple:
        """Sample the qubo problem.

        Args:
            init_sample (list): initial sample for the optimization
            Tschedule (list): temperature schedule for the optimization
            save_traj (bool, optional): save the trajectory. Defaults to False.
            verbose (bool, optional): print status. Defaults to False.

        Returns:
            Tuple: Solver status, str, solution, SimulatedAnnealingResults
        """
        res = self.sampler.sample(
            self.qubo,
            init_sample=init_sample,
            Tschedule=Tschedule,
            take_step=self.step_func,
            save_traj=save_traj,
            verbose=verbose,
        )

        # extact the solution and convert it
        idx_min = np.array([e for e in res.energies]).argmin()
        # idx_min = -1
        sol = res.trajectory[idx_min]
        sol = self.qubo.decode_solution(np.array(sol))
        sol = self.combine_flow_values(sol)
        sol = self.convert_solution_to_si(sol)

        # remove the height of the junction
        for i in range(self.wn.num_junctions):
            sol[self.wn.num_pipes + i] -= self.wn.nodes[
                self.wn.junction_name_list[i]
            ].elevation

        # load data in the AML model
        self.model.set_structure()
        self.load_data_in_model(self.model, sol)

        # returns
        return (SolverStatus.converged, "Solved Successfully", sol, res)
