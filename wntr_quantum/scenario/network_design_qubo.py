from wntr.sim import aml
from wntr.sim.models import constants
from wntr.sim.models import constraint
from wntr.sim.models import param
from wntr.sim.models import var
from wntr.sim.models.utils import ModelUpdater
from .chezy_manning import approx_chezy_manning_headloss_constraint
from .chezy_manning import chezy_manning_constants
from .chezy_manning import cm_resistance_param


class NetworkDesign(object):
    """Design problem solved using a QUBO approach."""

    def __init__(self, wn):
        """_summary_.

        Args:
            wn (_type_): _description_
        """
        self.wn = wn
        self.m, self.model_updater = self.create_cm_model()

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
