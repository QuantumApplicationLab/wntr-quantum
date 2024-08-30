from wntr.sim import aml
from wntr.sim.models import constants
from wntr.sim.models import constraint
from wntr.sim.models import param
from wntr.sim.models import var
from wntr.sim.models.utils import ModelUpdater
from .models.chezy_manning import approx_chezy_manning_headloss_constraint
from .models.chezy_manning import chezy_manning_constants
from .models.chezy_manning import cm_resistance_param
from .models.darcy_weisbach import approx_darcy_weisbach_headloss_constraint
from .models.darcy_weisbach import darcy_weisbach_constants
from .models.darcy_weisbach import dw_resistance_param


def create_hydraulic_model(wn):
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
    if wn.options.hydraulic.demand_model in ["PDD", "PDA"]:
        raise ValueError("Pressure Driven simulations not supported")

    if wn.options.hydraulic.headloss == "C-M":
        import_constants = chezy_manning_constants
        resistance_param = cm_resistance_param
        approx_head_loss_constraint = approx_chezy_manning_headloss_constraint
    elif wn.options.hydraulic.headloss == "D-W":
        import_constants = darcy_weisbach_constants
        resistance_param = dw_resistance_param
        approx_head_loss_constraint = approx_darcy_weisbach_headloss_constraint
    else:
        raise ValueError(
            "QUBO Hydraulic Simulations only supported for C-M and D-W simulations"
        )

    m = aml.Model()
    model_updater = ModelUpdater()

    # Global constants
    import_constants(m)
    constants.head_pump_constants(m)
    constants.leak_constants(m)
    constants.pdd_constants(m)

    param.source_head_param(m, wn)
    param.expected_demand_param(m, wn)

    param.leak_coeff_param.build(m, wn, model_updater)
    param.leak_area_param.build(m, wn, model_updater)
    param.leak_poly_coeffs_param.build(m, wn, model_updater)
    param.elevation_param.build(m, wn, model_updater)

    resistance_param.build(m, wn, model_updater)
    param.minor_loss_param.build(m, wn, model_updater)
    param.tcv_resistance_param.build(m, wn, model_updater)
    param.pump_power_param.build(m, wn, model_updater)
    param.valve_setting_param.build(m, wn, model_updater)

    var.flow_var(m, wn)
    var.head_var(m, wn)
    var.leak_rate_var(m, wn)

    constraint.mass_balance_constraint.build(m, wn, model_updater)

    approx_head_loss_constraint.build(m, wn, model_updater)

    constraint.head_pump_headloss_constraint.build(m, wn, model_updater)
    constraint.power_pump_headloss_constraint.build(m, wn, model_updater)
    constraint.prv_headloss_constraint.build(m, wn, model_updater)
    constraint.psv_headloss_constraint.build(m, wn, model_updater)
    constraint.tcv_headloss_constraint.build(m, wn, model_updater)
    constraint.fcv_headloss_constraint.build(m, wn, model_updater)
    if len(wn.pbv_name_list) > 0:
        raise NotImplementedError(
            "PBV valves are not currently supported in the WNTRSimulator"
        )
    if len(wn.gpv_name_list) > 0:
        raise NotImplementedError(
            "GPV valves are not currently supported in the WNTRSimulator"
        )
    constraint.leak_constraint.build(m, wn, model_updater)

    # TODO: Document that changing a curve with controls does not do anything; you have to change the pump_curve_name attribute on the pump

    return m, model_updater
