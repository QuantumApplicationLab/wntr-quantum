import itertools
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
from wntr.sim import aml
from wntr.sim.aml import Model
from wntr.sim.solvers import SolverStatus
from ..sim.models.chezy_manning import cm_resistance_value
from ..sim.models.chezy_manning import get_pipe_design_chezy_manning_qubops_matrix
from ..sim.models.darcy_weisbach import dw_resistance_value
from ..sim.models.darcy_weisbach import get_pipe_design_darcy_wesibach_qubops_matrix
from ..sim.models.mass_balance import get_mass_balance_qubops_matrix
from ..sim.qubo_hydraulics import create_hydraulic_model_for_qubo


class QUBODesignPipeDiameter(object):
    """Design problem solved using a QUBO approach."""

    def __init__(
        self,
        wn: WaterNetworkModel,
        flow_encoding: BaseQbitEncoding,
        head_encoding: BaseQbitEncoding,
        pipe_diameters: List,
        head_lower_bound: float,
        weight_cost: float = 1e-1,
    ):  # noqa: D417
        """Initialize the designer object.

        Args:
            wn (WaterNetworkModel): Water network
            flow_encoding (BaseQbitEncoding): binary encoding for the flows
            head_encoding (BaseQbitEncoding): binary encoding for the heads
            pipe_diameters (List): List of pipe diameters in SI
            head_lower_bound (float): minimum value for the head pressure values (US units)
            weight_cost (float, optional): weight for the cost optimization. Defaults to 1e-1.
        """
        # water network
        self.wn = wn

        # pipe diameters (converts to meter)
        self.pipe_diameters = [p / 1000 for p in pipe_diameters]
        self.num_diameters = len(pipe_diameters)

        # create the encoding vectors for the sign of the flows
        self.sign_flow_encoding = PositiveQbitEncoding(
            nqbit=1, step=2, offset=-1, var_base_name="x"
        )

        # create the solution vector for the sign
        self.sol_vect_signs = SolutionVector(
            wn.num_pipes, encoding=self.sign_flow_encoding
        )

        # store the flow encoding and create solution vector
        self.flow_encoding = flow_encoding
        self.sol_vect_flows = SolutionVector(wn.num_pipes, encoding=flow_encoding)
        if np.min(self.flow_encoding.get_possible_values()) < 0:
            raise ValueError(
                "The encoding of the flows must only take positive values."
            )

        # store the heqd encoding and create solution vector
        self.head_encoding = head_encoding
        self.sol_vect_heads = SolutionVector(wn.num_junctions, encoding=head_encoding)

        # one hot encoding for the pipe coefficients
        self.num_hot_encoding = wn.num_pipes * self.num_diameters
        self.pipe_encoding = PositiveQbitEncoding(1, "x_", offset=0, step=1)
        self.sol_vect_pipes = SolutionVector(self.num_hot_encoding, self.pipe_encoding)

        # mixed solution vector
        self.mixed_solution_vector = MixedSolutionVector(
            [
                self.sol_vect_signs,
                self.sol_vect_flows,
                self.sol_vect_heads,
                self.sol_vect_pipes,
            ]
        )

        # basic hydraulic model
        self.model, self.model_updater = create_hydraulic_model_for_qubo(wn)

        # valies of the pipe diameters/coefficients
        self.get_pipe_data()

        # weight for the cost equation
        self.weight_cost = weight_cost

        # lower bound for the pressure
        self.head_lower_bound = head_lower_bound
        self.head_upper_bound = 10 * head_lower_bound  # is that enough ?

        # store other attributes
        self.qubo = None
        self.flow_index_mapping = None
        self.head_index_mapping = None

    def get_dw_pipe_coefficients(self, link):
        """Get the pipe coefficients for a specific link with DW.

        Args:
            link (_type_): _description_
        """
        values = []
        for diam in self.pipe_diameters:

            # convert values from SI to epanet internal
            roughness_us = 0.001 * from_si(
                FlowUnits.CFS, link.roughness, HydParam.Length
            )
            diameter_us = from_si(FlowUnits.CFS, diam, HydParam.Length)
            length_us = from_si(FlowUnits.CFS, link.length, HydParam.Length)

            # compute the resistance value fit coefficients
            values.append(
                dw_resistance_value(
                    self.model.dw_k,
                    roughness_us,
                    diameter_us,
                    self.model.dw_diameter_exp,
                    length_us,
                )
            )
        return values

    def get_cm_pipe_coefficients(self, link):
        """Get the pipe coefficients for a specific link with CM.

        Args:
            link (_type_): _description_
        """
        values = []
        for diam in self.pipe_diameters:

            # convert values from SI to epanet internal
            roughness_us = link.roughness
            diameter_us = from_si(FlowUnits.CFS, diam, HydParam.Length)
            length_us = from_si(FlowUnits.CFS, link.length, HydParam.Length)

            # compute the resistance value fit coefficients
            values.append(
                cm_resistance_value(
                    self.model.cm_k,
                    roughness_us,
                    self.model.cm_roughness_exp,
                    diameter_us,
                    self.model.cm_diameter_exp,
                    length_us,
                )
            )
        return values

    def get_pipe_prices(self, link):
        """Get the price of the pipe for the different diameters.

        Args:
            link (wn.link): pipe info
        """

        def _compute_price(diameter, length):
            """Price model of the pipe.

            Args:
                diameter (float): diameter
                length (float): length

            Returns:
                float: price
            """
            return np.pi * diameter * length / 1e5

        prices = []
        for diam in self.pipe_diameters:

            # convert values from SI to epanet internal
            diameter_us = from_si(FlowUnits.CFS, diam, HydParam.Length)
            length_us = from_si(FlowUnits.CFS, link.length, HydParam.Length)

            # compute the price
            prices.append(_compute_price(diameter_us, length_us))

        return prices

    def get_pipe_data(self):
        """Get the parameters of the AML model related to each pipe.

        Returns:
            Dict: possible pipe coefficients for each coefficients
        """
        if not hasattr(self.model, "pipe_coefficients"):
            self.model.pipe_coefficients = aml.ParamDict()

        if not hasattr(self.model, "pipe_coefficients_indices"):
            self.model.pipe_coefficients_indices = aml.ParamDict()

        if not hasattr(self.model, "pipe_prices"):
            self.model.pipe_prices = aml.ParamDict()

        # select model
        if self.wn.options.hydraulic.headloss == "C-M":
            get_pipe_coeff_values = self.get_cm_pipe_coefficients

        elif self.wn.options.hydraulic.headloss == "D-W":
            get_pipe_coeff_values = self.get_dw_pipe_coefficients

        # loop over pipes
        idx_start = 0
        for link_name in self.wn.pipe_name_list:

            # get the link
            link = self.wn.get_link(link_name)

            # compute the pipe coeffcient values
            pipe_coeffs_values = get_pipe_coeff_values(link)
            if link_name in self.model.pipe_coefficients:
                self.model.pipe_coefficients[link_name].value = pipe_coeffs_values
            else:
                self.model.pipe_coefficients[link_name] = aml.Param(pipe_coeffs_values)

            # compute the pipe price
            prices = self.get_pipe_prices(link)
            if link_name in self.model.pipe_prices:
                self.model.pipe_prices[link_name].value = prices
            else:
                self.model.pipe_prices[link_name] = aml.Param(prices)

            # compute the indices
            idx_end = idx_start + len(pipe_coeffs_values)
            indices = list(range(idx_start, idx_end))
            if link_name in self.model.pipe_coefficients_indices:
                self.model.pipe_coefficients_indices[link_name].value = indices
            else:
                self.model.pipe_coefficients_indices[link_name] = aml.Param(indices)
            idx_start = len(pipe_coeffs_values)

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

    def verify_solution(self, input, params):
        """Computes the rhs vector associate with the input.

        Args:
            input (np.ndarray): proposed solution vector
            params (list): one-hot encoding vector to select the resistance factor.

        Returns:
            np.ndarray: RHS vector
        """
        P0, P1, P2, P3, P4 = self.matrices
        num_heads = self.wn.num_junctions
        num_signs = self.wn.num_pipes
        num_pipes = self.wn.num_pipes
        num_vars = num_heads + 2 * num_pipes

        input = input.reshape(-1, 1)
        p0 = P0[:-1].reshape(-1, 1)
        p1 = P1[:-1, num_signs:num_vars] + P2.sum(1)[:-1, num_signs:num_vars]
        p2 = P4.sum(1)[:-1, num_pipes:num_vars, num_pipes:num_vars].sum(-2)
        parameters = np.array([0] * num_vars + params)
        p2 = (parameters * p2).sum(-1)
        sign = np.sign(input)
        return p0 + p1 @ input + (p2 @ (sign * input * input))

    def enumerates_classical_solutions(self, convert_to_si=True):
        """Generates the classical solution."""
        encoding = []
        for idiam in range(self.num_diameters):
            tmp = [0] * self.num_diameters
            tmp[idiam] = 1
            encoding.append(tmp)

        print("price \t diameters \t variables")
        for params in itertools.product(encoding, repeat=self.wn.num_pipes):
            pvalues = []
            for p in params:
                pvalues += p
            price, diameters = self.get_pipe_info_from_hot_encoding(pvalues)
            sol = self.compute_classical_solution(pvalues, convert_to_si=convert_to_si)
            print(price, diameters, sol)

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

    def compute_classical_solution(self, parameters, convert_to_si=True):
        """Computes the classical solution for a values of the hot encoding parameters.

        Args:
            parameters (List): list of the one hot encoding values e.g. [1,0,1,0]
            convert_to_si (bool): convert to si

        Returns:
            np.mdarray : solution
        """
        P0, P1, P2, P3, P4 = self.matrices
        num_heads = self.wn.num_junctions
        num_signs = self.wn.num_pipes
        num_pipes = self.wn.num_pipes
        num_vars = num_heads + 2 * num_pipes

        if self.wn.options.hydraulic.headloss == "C-M":
            p0 = P0[:-1].reshape(-1, 1)
            p1 = P1[:-1, num_signs:num_vars] + P2.sum(1)[:-1, num_signs:num_vars]
            p2 = P4.sum(1)[:-1, num_pipes:num_vars, num_pipes:num_vars].sum(-2)
            parameters = np.array([0] * num_vars + parameters)
            p2 = (parameters * p2).sum(-1)

        elif self.wn.options.hydraulic.headloss == "D-W":

            # P0 matrix
            p0 = np.copy(P0[:-1])
            # add the k0 term without sgin  to p0
            p0 += (
                (parameters * P2[:-1, :num_signs, num_vars:])
                .sum(-1)
                .sum(-1)
                .reshape(-1, 1)
            )

            # P1 matrix
            p1 = P1[:-1, num_pipes:num_vars] + P2.sum(1)[:-1, num_pipes:num_vars]

            # add the terms in k1
            p1 += (
                (parameters * P3[:-1, :num_signs, num_signs:num_vars, num_vars:])
                .sum(1)
                .sum(-1)
            )

            # P2 matrix
            p2 = (
                (
                    parameters
                    * P4[
                        :-1,
                        :num_signs,
                        num_signs:num_vars,
                        num_signs:num_vars,
                        num_vars:,
                    ]
                )
                .sum(1)
                .sum(-1)
                .sum(-1)
            )

        def func(input):
            input = input.reshape(-1, 1)
            sign = np.sign(input)
            sol = p0 + p1 @ input + (p2 @ (sign * input * input))
            return sol.reshape(-1)

        initial_point = np.random.rand(num_pipes + num_heads)
        res = newton_raphson(func, initial_point)
        assert np.allclose(func(res.solution), 0)
        if convert_to_si:
            return self.convert_solution_to_si(res.solution)
        return res.solution

    def get_cost_matrix(self, matrices):
        """Add the equation that ar sued to maximize the pipe coefficiens and therefore minimize the diameter.

        Args:
            matrices (tuple): The matrices
        """
        P0, P1, P2, P3, P4 = matrices

        # loop over all the pipe coeffs
        istart = 2 * self.sol_vect_flows.size + self.sol_vect_heads.size
        index_over = self.wn.pipe_name_list

        # loop over all the pipe coeffs
        for link_name in index_over:
            for pipe_cost, pipe_idx in zip(
                self.model.pipe_prices[link_name].value,
                self.model.pipe_coefficients_indices[link_name].value,
            ):
                P1[-1, istart + pipe_idx] = self.weight_cost * pipe_cost
        return P0, P1, P2, P3, P4

    def initialize_matrices(self) -> Tuple:
        """Creates the matrices of the polynomial system of equation.

        math ::


        """
        num_equations = len(list(self.model.cons())) + 1  # +1 for cost equation
        num_variables = (
            2 * len(self.model.flow) + len(self.model.head) + self.num_hot_encoding
        )

        # must transform that to coo
        P0 = np.zeros((num_equations, 1))
        P1 = np.zeros((num_equations, num_variables))
        P2 = np.zeros((num_equations, num_variables, num_variables))
        P3 = np.zeros((num_equations, num_variables, num_variables, num_variables))
        P4 = np.zeros(
            (num_equations, num_variables, num_variables, num_variables, num_variables)
        )

        # get the mass balance matrix
        (P0, P1, P2, P3) = get_mass_balance_qubops_matrix(
            self.model,
            self.wn,
            (P0, P1, P2, P3),
            self.flow_index_mapping,
            convert_to_us_unit=True,
        )

        # shortcut
        matrices = (P0, P1, P2, P3, P4)

        # get the headloss matrix contributions
        if self.wn.options.hydraulic.headloss == "C-M":
            matrices = get_pipe_design_chezy_manning_qubops_matrix(
                self.model,
                self.wn,
                matrices,
                self.flow_index_mapping,
                self.head_index_mapping,
            )
        elif self.wn.options.hydraulic.headloss == "D-W":
            matrices = get_pipe_design_darcy_wesibach_qubops_matrix(
                self.model,
                self.wn,
                matrices,
                self.flow_index_mapping,
                self.head_index_mapping,
            )
        else:
            raise ValueError("Calculation only possible with C-M or D-W")

        matrices = self.get_cost_matrix(matrices)

        return matrices

    @staticmethod
    def flatten_solution_vector(solution: Tuple) -> List:
        """Flattens the solution vector.

        Args:
            solution (tuple): tuple of ([flows], [heads])

        Returns:
            List: a flat list of all the variables
        """
        sol_tmp = []
        for s in solution[:-1]:
            sol_tmp += s
        return sol_tmp, solution[-1]

    def get_pipe_info_from_hot_encoding(self, hot_encoding):
        """_summary_.

        Args:
            hot_encoding (_type_): _description_
        """
        hot_encoding = np.array(hot_encoding)

        pipe_prices = []
        for link_name in self.wn.pipe_name_list:
            pipe_prices += self.model.pipe_prices[link_name].value
        pipe_prices = np.array(pipe_prices)
        total_price = (pipe_prices * hot_encoding).sum()

        pipe_diameters = 1000 * np.array(self.pipe_diameters * self.wn.num_pipes)
        pipe_diameters = (
            (pipe_diameters * hot_encoding).reshape(-1, self.num_diameters).sum(-1)
        )

        return total_price, pipe_diameters

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

    def create_index_mapping(self) -> None:
        """Creates the index maping for qubops matrices."""
        # init the idx
        idx = 0

        # number of variables that are flows
        num_flow_var = len(self.model.flow)
        num_head_var = len(self.model.head)

        # get the indices for the sign/abs value of the flow
        self.flow_index_mapping = OrderedDict()
        for _, val in self.model.flow.items():
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
        for _, val in self.model.head.items():
            self.head_index_mapping[val.name] = 2 * num_flow_var + idx
            idx += 1

        # get the indices for the pipe diameters
        idx = 0
        self.pipe_diameter_index_mapping = OrderedDict()
        for _, val in self.model.flow.items():
            if val.name not in self.pipe_diameter_index_mapping:
                self.pipe_diameter_index_mapping[val.name] = OrderedDict()
                for idiam in range(self.num_diameters):
                    self.pipe_diameter_index_mapping[val.name][idiam] = (
                        2 * num_flow_var + num_head_var + idx
                    )
                    idx += 1

    def solve(  # noqa: D417
        self, strength: float = 1e6, num_reads: int = 10000, **options
    ) -> Tuple:
        """Solves the Hydraulics equations.

        Args:
            strength (float, optional): substitution strength. Defaults to 1e6.
            num_reads (int, optional): number of reads for the sampler. Defaults to 10000.

        Returns:
            Tuple: Succes message
        """
        # create the index mapping of the variables
        self.create_index_mapping()

        # compute the polynomial matrices
        self.matrices = self.initialize_matrices()

        self.qubo = QUBOPS_MIXED(self.mixed_solution_vector, **options)
        matrices = tuple(sparse.COO(m) for m in self.matrices)

        # create the BQM
        self.bqm = self.qubo.create_bqm(matrices, strength=strength)

        # add constraints on the hot encoding
        # the sum of each hot encoding variable of a given pipe must equals 1
        istart = (
            self.sol_vect_signs.size
            + self.sol_vect_flows.size
            + self.sol_vect_heads.size
        )
        for i in range(self.sol_vect_flows.size):

            # create the expression [(x0, 1), (x1, 1), ...]
            expr = []
            iend = istart + self.num_diameters
            for ivar in range(istart, iend):
                expr.append(
                    (
                        self.mixed_solution_vector.encoded_reals[ivar]
                        .variables[0]
                        .name,
                        1,
                    )
                )
            # add the constraints
            self.bqm.add_linear_equality_constraint(
                expr, lagrange_multiplier=strength, constant=-1
            )
            istart += self.num_diameters

        # add constraint on head pressures
        istart = 2 * self.sol_vect_flows.size
        for i in range(self.sol_vect_heads.size):

            self.bqm.add_linear_inequality_constraint(
                self.qubo.all_expr[istart + i],
                lagrange_multiplier=1,
                label="head_%s" % i,
                lb=self.head_lower_bound,
                ub=self.head_upper_bound,
            )

        # sample
        sampleset = self.qubo.sample_bqm(self.bqm, num_reads=num_reads)

        # decode
        sol = self.qubo.decode_solution(sampleset.lowest().record[0][0])

        # flatten
        sol, hot_encoding = self.flatten_solution_vector(sol)
        print(sol)
        # convert back to SI
        sol = self.convert_solution_to_si(sol)

        # load data in the AML model
        self.model.set_structure()
        self.load_data_in_model(self.model, sol)

        # get pipe info from one hot
        self.total_pice, self.optimal_diameters = self.get_pipe_info_from_hot_encoding(
            hot_encoding
        )

        # returns
        return (
            SolverStatus.converged,
            "Solved Successfully",
            0,
        )
