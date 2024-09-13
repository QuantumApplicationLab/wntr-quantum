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
        m (aml.Model): Model of the netwwork
    """
    m.dw_k = 0.025173  # 16/64.4/pi^2
    m.dw_exp = 2
    m.dw_diameter_exp = -5


def dw_resistance_prefactor(k, roughness, diameter, diameter_exp):
    """Computes the resistance prefactor.

    Args:
        k (float): scaling parameter of the approximatioj
        roughness (float): roughness pf the pipe
        diameter (float): dimater of the pipe
        diameter_exp (int): exponent of the pip diameter in the approx (typically -5)

    Returns:
        Tuple(float, float, float): value of the fit to the full DW formula
    """
    return (
        k
        * (diameter**diameter_exp)
        * dw_fit(roughness, diameter, convert_to_us_unit=False)
    )


def dw_resistance_value(k, roughness, diameter, diameter_exp, length):
    """_summary_.

    Args:
        k (float): scaling parameter of the approximatioj
        roughness (float): roughness pf the pipe
        diameter (float): dimater of the pipe
        diameter_exp (int): exponent of the pip diameter in the approx (typically -5)
        length (float): length of the pipe

    Returns:
        Tuple(float, float, float): value of the fit to the full DW formula
    """
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


def get_darcy_weisbach_qubops_matrix(
    m, wn, matrices, flow_index_mapping, head_index_mapping
):  # noqa: D417
    """Create the matrices for chezy manning headloss approximation.

    Args:
        m (aml.Model): The AML model of the network
        wn (WaternNetwork): th water network object
        matrices (Tuple): The qubops matrices of the network
        flow_index_mapping (Dict): A dict to map the flow model variables to the qubops matrices
        head_index_mapping (Dict): A dict to map the head model variables to the qubops matrices
        convert_to_us_unit (bool, optional): Convert the inut to US units. Defaults to False.

    Returns:
        Tuple: The qubops matrices of the network
    """
    P0, P1, P2, P3 = matrices

    for ieq0, link_name in enumerate(wn.pipe_name_list):

        # index of the pipe equation
        ieq = ieq0 + len(wn.junction_name_list)

        # get link info
        link = wn.get_link(link_name)

        # get start/end node info
        start_node_name = link.start_node_name
        end_node_name = link.end_node_name
        start_node = wn.get_node(start_node_name)
        end_node = wn.get_node(end_node_name)

        # linear term (start head values) of the headloss approximation
        if isinstance(start_node, wntr.network.Junction):
            start_node_index = head_index_mapping[m.head[start_node_name].name]
            P1[ieq, start_node_index] += 1
        else:
            start_h = m.source_head[start_node_name]
            P0[ieq, 0] += from_si(FlowUnits.CFS, start_h.value, HydParam.Length)

        # linear term (end head values) of the headloss approximation
        if isinstance(end_node, wntr.network.Junction):
            end_node_index = head_index_mapping[m.head[end_node_name].name]
            P1[ieq, end_node_index] -= 1
        else:
            end_h = m.source_head[end_node_name]
            P0[ieq, 0] -= from_si(FlowUnits.CFS, end_h.value, HydParam.Length)

        # non linear term (sign flow^2) of headloss approximation
        k0 = m.dw_resistance_0[link_name]
        k1 = m.dw_resistance_1[link_name]
        k2 = m.dw_resistance_2[link_name]

        sign_index = flow_index_mapping[m.flow[link_name].name]["sign"]
        flow_index = flow_index_mapping[m.flow[link_name].name]["absolute_value"]

        P0[ieq] -= k0.value
        P2[ieq, sign_index, flow_index] -= k1.value
        P3[ieq, sign_index, flow_index, flow_index] -= k2.value

    return (P0, P1, P2, P3)
