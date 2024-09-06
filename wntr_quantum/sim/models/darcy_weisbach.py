import wntr
from wntr.epanet.util import FlowUnits
from wntr.epanet.util import HydParam
from wntr.epanet.util import from_si
from wntr.network import LinkStatus
from wntr.sim import aml
from wntr.sim.models.utils import Definition
from .darcy_weisbach_fit import dw_fit


def darcy_weisbach_constants(m):
    """Add darcy weisbach constants to the model.

    Args:
        m (_type_): _description_
    """
    m.dw_k = 0.025173  # 16/64.4/pi^2
    m.dw_exp = 2
    m.dw_diameter_exp = -5


def dw_resistance_prefactor(k, roughness, diameter, diameter_exp):
    """_summary_.

    Args:
        k (_type_): _description_
        roughness (_type_): _description_
        exp (_type_): _description_
        diameter (_type_): _description_
        diameter_exp (_type_): _description_
    """
    return (
        k
        * (diameter**diameter_exp)
        * dw_fit(roughness, diameter, convert_to_us_unit=False)
    )


def dw_resistance_value(k, roughness, diameter, diameter_exp, length):
    """_summary_.

    Args:
        k (_type_): _description_
        roughness (_type_): _description_
        exp (_type_): _description_
        diameter (_type_): _description_
        diameter_exp (_type_): _description_
        length (_type_): _description_

    Returns:
        _type_: _description_
    """
    # print("Roughness : %f" % roughness)
    # print("diameter : %f" % diameter)
    # print("resistance coeff : %f " % (k * (diameter**diameter_exp) * length))
    return dw_resistance_prefactor(k, roughness, diameter, diameter_exp) * length


class dw_resistance_param(Definition):  # noqa: D101
    @classmethod
    def build(cls, m, wn, updater, index_over=None):  # noqa: D417
        """Add a CM resistance coefficient parameter to the model.

        Parameters
        ----------
        m: wntr.aml.aml.aml.Model
        wn: wntr.network.model.WaterNetworkModel
        updater: ModelUpdater
        index_over: list of str
            list of pipe names
        """
        if not hasattr(m, "dw_resistance_0"):
            m.dw_resistance_0 = aml.ParamDict()
        if not hasattr(m, "dw_resistance_1"):
            m.dw_resistance_1 = aml.ParamDict()
        if not hasattr(m, "dw_resistance_2"):
            m.dw_resistance_2 = aml.ParamDict()

        if index_over is None:
            index_over = wn.pipe_name_list

        for link_name in index_over:
            link = wn.get_link(link_name)

            # convert values from SI to epanet internal
            roughness_us = 0.001 * from_si(
                FlowUnits.CFS, link.roughness, HydParam.Length
            )
            diameter_us = from_si(FlowUnits.CFS, link.diameter, HydParam.Length)
            length_us = from_si(FlowUnits.CFS, link.length, HydParam.Length)

            # compute the resistance value fit coefficients
            value = dw_resistance_value(
                m.dw_k,
                roughness_us,
                diameter_us,
                m.dw_diameter_exp,
                length_us,
            )
            if link_name in m.dw_resistance_0:
                m.dw_resistance_0[link_name].value = value[0]
            else:
                m.dw_resistance_0[link_name] = aml.Param(value[0])

            if link_name in m.dw_resistance_1:
                m.dw_resistance_1[link_name].value = value[1]
            else:
                m.dw_resistance_1[link_name] = aml.Param(value[1])

            if link_name in m.dw_resistance_2:
                m.dw_resistance_2[link_name].value = value[2]
            else:
                m.dw_resistance_2[link_name] = aml.Param(value[2])

            updater.add(link, "roughness", dw_resistance_param.update)
            updater.add(link, "diameter", dw_resistance_param.update)
            updater.add(link, "length", dw_resistance_param.update)


class approx_darcy_weisbach_headloss_constraint(Definition):  # noqa: D101
    @classmethod
    def build(cls, m, wn, updater, index_over=None):  # noqa: D417
        """Adds a mass balance to the model for the specified junctions.

        Parameters
        ----------
        m: wntr.aml.aml.aml.Model
        wn: wntr.network.model.WaterNetworkModel
        updater: ModelUpdater
        index_over: list of str
            list of pipe names; default is all pipes in wn
        """
        if not hasattr(m, "approx_darcy_weisbach_headloss"):
            m.approx_darcy_wesibach_headloss = aml.ConstraintDict()

        if index_over is None:
            index_over = wn.pipe_name_list

        for link_name in index_over:
            if link_name in m.approx_darcy_wesibach_headloss:
                del m.approx_darcy_wesibach_headloss[link_name]

            link = wn.get_link(link_name)
            f = m.flow[link_name]
            status = link.status

            if status == LinkStatus.Closed or link._is_isolated:
                con = aml.Constraint(f)
            else:
                start_node_name = link.start_node_name
                end_node_name = link.end_node_name
                start_node = wn.get_node(start_node_name)
                end_node = wn.get_node(end_node_name)
                if isinstance(start_node, wntr.network.Junction):
                    start_h = m.head[start_node_name]
                else:
                    start_h = m.source_head[start_node_name]
                if isinstance(end_node, wntr.network.Junction):
                    end_h = m.head[end_node_name]
                else:
                    end_h = m.source_head[end_node_name]
                k0 = m.dw_resistance_0[link_name]
                k1 = m.dw_resistance_1[link_name]
                k2 = m.dw_resistance_2[link_name]

                con = aml.Constraint(expr=-k0 - k1 * f - k2 * f**2 + start_h - end_h)

            m.approx_darcy_wesibach_headloss[link_name] = con

            updater.add(
                link, "status", approx_darcy_weisbach_headloss_constraint.update
            )
            updater.add(
                link, "_is_isolated", approx_darcy_weisbach_headloss_constraint.update
            )


def get_darcy_weisbach_matrix(m, wn, matrices):  # noqa: D417
    """Adds a mass balance to the model for the specified junctions.

    Parameters
    ----------
    m: wntr.aml.aml.aml.Model
    wn: wntr.network.model.WaterNetworkModel
    updater: ModelUpdater
    index_over: list of str
        list of pipe names; default is all pipes in wn
    """
    P0, P1, P2 = matrices

    continuous_var_name = [v.name for v in list(m.vars())]
    # discrete_var_name = [v.name for k, v in m.dw_resistance.items()]

    var_names = continuous_var_name  # + discrete_var_name

    index_over = wn.pipe_name_list

    for ieq0, link_name in enumerate(index_over):

        ieq = ieq0 + len(wn.junction_name_list)
        link = wn.get_link(link_name)
        f = m.flow[link_name]
        flow_index = var_names.index(f.name)

        start_node_name = link.start_node_name
        end_node_name = link.end_node_name

        start_node = wn.get_node(start_node_name)
        end_node = wn.get_node(end_node_name)

        if isinstance(start_node, wntr.network.Junction):
            start_h = m.head[start_node_name]
            start_node_index = var_names.index(start_h.name)
            P1[ieq, start_node_index] = 1
        else:
            start_h = m.source_head[start_node_name]
            P0[ieq, 0] += from_si(FlowUnits.CFS, start_h.value, HydParam.Length)

        if isinstance(end_node, wntr.network.Junction):
            end_h = m.head[end_node_name]
            end_node_index = var_names.index(end_h.name)
            P1[ieq, end_node_index] = -1
        else:
            end_h = m.source_head[end_node_name]
            P0[ieq, 0] -= from_si(FlowUnits.CFS, end_h.value, HydParam.Length)

        k0 = m.dw_resistance_0[link_name]
        k1 = m.dw_resistance_1[link_name]
        k2 = m.dw_resistance_2[link_name]
        # print(k0.value, k1.value, k2.value)

        P0[ieq] -= k0.value
        P1[ieq, flow_index] -= k1.value
        P2[ieq, flow_index, flow_index] -= k2.value

    return (P0, P1, P2)


def get_pipe_design_darcy_wesibach_matrix(m, wn, matrices):  # noqa: D417
    """Adds a mass balance to the model for the specified junctions.

    Parameters
    ----------
    m: wntr.aml.aml.aml.Model
    wn: wntr.network.model.WaterNetworkModel
    updater: ModelUpdater
    index_over: list of str
        list of pipe names; default is all pipes in wn
    """
    P0, P1, P2, P3 = matrices

    continuous_var_name = [v.name for v in list(m.vars())]
    num_continuous_var = len(continuous_var_name)
    # discrete_var_name = [v.name for k, v in m.cm_resistance.items()]
    var_names = continuous_var_name  # + discrete_var_name

    index_over = wn.pipe_name_list

    for ieq0, link_name in enumerate(index_over):

        ieq = ieq0 + len(wn.junction_name_list)
        link = wn.get_link(link_name)
        f = m.flow[link_name]
        flow_index = var_names.index(f.name)

        start_node_name = link.start_node_name
        end_node_name = link.end_node_name

        start_node = wn.get_node(start_node_name)
        end_node = wn.get_node(end_node_name)

        if isinstance(start_node, wntr.network.Junction):
            start_h = m.head[start_node_name]
            start_node_index = var_names.index(start_h.name)
            P1[ieq, start_node_index] = 1
        else:
            start_h = m.source_head[start_node_name]
            P0[ieq, 0] += from_si(FlowUnits.CFS, start_h.value, HydParam.Length)

        if isinstance(end_node, wntr.network.Junction):
            end_h = m.head[end_node_name]
            end_node_index = var_names.index(end_h.name)
            P1[ieq, end_node_index] = -1
        else:
            end_h = m.source_head[end_node_name]
            P0[ieq, 0] -= from_si(FlowUnits.CFS, end_h.value, HydParam.Length)

        for pipe_coefs, pipe_idx in zip(
            m.pipe_coefficients[link_name].value,
            m.pipe_coefficients_indices[link_name].value,
        ):
            P1[ieq, pipe_idx + num_continuous_var] -= pipe_coefs[0]
            P2[ieq, flow_index, pipe_idx + num_continuous_var] -= pipe_coefs[1]
            P3[
                ieq, flow_index, flow_index, pipe_idx + num_continuous_var
            ] -= -pipe_coefs[2]

    return (P0, P1, P2, P3)
