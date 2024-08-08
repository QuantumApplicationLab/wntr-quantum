import itertools
import numpy as np
from quantum_newton_raphson.newton_raphson import newton_raphson
from qubols.encodings import DiscreteValuesEncoding
from qubols.mixed_solution_vector import MixedSolutionVector_V2 as MixedSolutionVector
from qubols.qubo_poly_mixed_variables import QUBO_POLY_MIXED
from qubols.solution_vector import SolutionVector_V2 as SolutionVector
from wntr.sim import aml
from wntr.sim.models import constants
from wntr.sim.models import constraint
from wntr.sim.models import param
from wntr.sim.models import var
from wntr.sim.models.utils import ModelUpdater
import sparse
from .chezy_manning import approx_chezy_manning_headloss_constraint
from .chezy_manning import chezy_manning_constants
from .chezy_manning import cm_resistance_param
from .chezy_manning import cm_resistance_prefactor
from .chezy_manning import get_chezy_manning_matrix
from .chezy_manning import get_mass_balance_constraint


class Network(object):
    """Design problem solved using a QUBO approach."""

    def __init__(
        self,
        wn,
        flow_encoding,
        head_encoding,
    ):  # noqa: D417
        """_summary_.

        Args:
            wn (_type_): _description_
            encoding_flows (_type_): _description_
            encoding_heads (_type_): _description_
            pipe_diameters (_type_): _description_
        """
        self.wn = wn
        self.sol_vect_flows = SolutionVector(wn.num_pipes, encoding=flow_encoding)
        self.sol_vect_heads = SolutionVector(
            wn.num_junctions, encoding=head_encoding
        )  # not sure num_junction is what we need

        self.m, self.model_updater = self.create_cm_model()

        self.mixed_solution_vector = MixedSolutionVector(
            [self.sol_vect_flows, self.sol_vect_heads]
        )

        self.matrices = self.initialize_matrices()

    def verify_solution(self, input):
        """generates the classical solution."""

        P0, P1, P2 = self.matrices

        p0 = P0.reshape(
            -1,
        )
        p1 = P1
        p2 = P2.sum(-1)
        return p0 + p1 @ input + (p2 @ (input * input))

    def classical_solutions(self):
        """generates the classical solution."""

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
        res = newton_raphson(func, initial_point)
        assert np.allclose(func(res.solution), 0)
        return res.solution

    def create_cm_model(self):
        """Create the aml.

        Args:
            wn (_type_): _description_

        Raises:
            NotImplementedError: _description_
            NotImplementedError: _description_
            ValueError: _description_
            ValueError: _description_
            NotImplementedError: _description_
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        if self.wn.options.hydraulic.demand_model in ["PDD", "PDA"]:
            raise ValueError("Pressure Driven simulations not supported")
        if self.wn.options.hydraulic.headloss not in ["C-M"]:
            raise ValueError("Quantum Design only supported for C-M simulations")

        m = aml.Model()
        model_updater = ModelUpdater()

        # Global constants
        chezy_manning_constants(m)
        constants.head_pump_constants(m)
        constants.leak_constants(m)
        constants.pdd_constants(m)

        param.source_head_param(m, self.wn)
        param.expected_demand_param(m, self.wn)

        param.leak_coeff_param.build(m, self.wn, model_updater)
        param.leak_area_param.build(m, self.wn, model_updater)
        param.leak_poly_coeffs_param.build(m, self.wn, model_updater)
        param.elevation_param.build(m, self.wn, model_updater)

        cm_resistance_param.build(m, self.wn, model_updater)
        param.minor_loss_param.build(m, self.wn, model_updater)
        param.tcv_resistance_param.build(m, self.wn, model_updater)
        param.pump_power_param.build(m, self.wn, model_updater)
        param.valve_setting_param.build(m, self.wn, model_updater)

        var.flow_var(m, self.wn)
        var.head_var(m, self.wn)
        var.leak_rate_var(m, self.wn)

        constraint.mass_balance_constraint.build(m, self.wn, model_updater)

        approx_chezy_manning_headloss_constraint.build(m, self.wn, model_updater)

        constraint.head_pump_headloss_constraint.build(m, self.wn, model_updater)
        constraint.power_pump_headloss_constraint.build(m, self.wn, model_updater)
        constraint.prv_headloss_constraint.build(m, self.wn, model_updater)
        constraint.psv_headloss_constraint.build(m, self.wn, model_updater)
        constraint.tcv_headloss_constraint.build(m, self.wn, model_updater)
        constraint.fcv_headloss_constraint.build(m, self.wn, model_updater)
        if len(self.wn.pbv_name_list) > 0:
            raise NotImplementedError(
                "PBV valves are not currently supported in the WNTRSimulator"
            )
        if len(self.wn.gpv_name_list) > 0:
            raise NotImplementedError(
                "GPV valves are not currently supported in the WNTRSimulator"
            )
        constraint.leak_constraint.build(m, self.wn, model_updater)

        # TODO: Document that changing a curve with controls does not do anything; you have to change the pump_curve_name attribute on the pump

        return m, model_updater

    def initialize_matrices(self):
        """_summary_."""
        num_equations = len(list(self.m.cons()))
        num_variables = len(list(self.m.vars()))

        # must transform that to coo
        P0 = np.zeros((num_equations, 1))
        P1 = np.zeros((num_equations, num_variables))
        P2 = np.zeros((num_equations, num_variables, num_variables))

        matrices = (P0, P1, P2)
        matrices = get_mass_balance_constraint(self.m, self.wn, matrices)
        matrices = get_chezy_manning_matrix(self.m, self.wn, matrices)

        return matrices

    def solve(self, **options):
        """_summary_"""
        qubo = QUBO_POLY_MIXED(self.mixed_solution_vector, **options)
        matrices = tuple(sparse.COO(m) for m in self.matrices)
        bqm = qubo.create_bqm(matrices, strength=1000)

        # sample
        sampleset = qubo.sample_bqm(bqm, num_reads=options["num_reads"])

        # decode
        sol, param = qubo.decode_solution(sampleset.lowest())
        return sol, param
