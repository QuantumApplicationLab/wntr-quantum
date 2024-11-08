import numpy as np
import wntr
from wntr.epanet.util import FlowUnits
from wntr.epanet.util import HydParam
from wntr.epanet.util import from_si
from wntr.network import LinkStatus
from wntr.sim import aml
from wntr.sim.models.utils import Definition


def chezy_manning_constants(m):
    """Add chezy manning constants to the model.

    Args:
        m (aml.Model): Model of the netwwork
    """
    m.cm_exp = 2
    m.cm_k = (4 / (1.49 * np.pi)) ** 2 * (1 / 4) ** -1.33
    m.cm_roughness_exp = 2
    m.cm_diameter_exp = -5.33


def cm_resistance_prefactor(k, roughness, roughness_exp, diameter, diameter_exp):
    """Computes the resistance prefactor.

    Args:
        k (float): scaling parameter of the approximatioj
        roughness (float): roughness pf the pipe
        roughness_exp(int): exponent of the pip diameter in the approx (typically 2)
        diameter (float): dimater of the pipe
        diameter_exp (int): exponent of the pip diameter in the approx (typically -5.33)

    Returns:
        float: resistance prefactor
    """
    return k * roughness ** (roughness_exp) * diameter ** (diameter_exp)


def cm_resistance_value(k, roughness, roughness_exp, diameter, diameter_exp, length):
    """Computes the resistance value.

    Args:
        k (float): scaling parameter of the approximatioj
        roughness (float): roughness pf the pipe
        roughness_exp(int): exponent of the pip diameter in the approx (typically 2)
        diameter (float): dimater of the pipe
        diameter_exp (int): exponent of the pip diameter in the approx (typically -5.33)
        length (float): length of the pipe

    Returns:
        float: resistance value
    """
    return (
        cm_resistance_prefactor(k, roughness, roughness_exp, diameter, diameter_exp)
        * length
    )


class cm_resistance_param(Definition):  # noqa: D101
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
        if not hasattr(m, "cm_resistance"):
            m.cm_resistance = aml.ParamDict()

        if index_over is None:
            index_over = wn.pipe_name_list

        for link_name in index_over:
            link = wn.get_link(link_name)

            # convert values from SI to epanet internal
            roughness_us = link.roughness
            diameter_us = from_si(FlowUnits.CFS, link.diameter, HydParam.Length)
            length_us = from_si(FlowUnits.CFS, link.length, HydParam.Length)

            value = cm_resistance_value(
                m.cm_k,
                roughness_us,
                m.cm_roughness_exp,
                diameter_us,
                m.cm_diameter_exp,
                length_us,
            )
            if link_name in m.cm_resistance:
                m.cm_resistance[link_name].value = value
            else:
                m.cm_resistance[link_name] = aml.Param(value)

            updater.add(link, "roughness", cm_resistance_param.update)
            updater.add(link, "diameter", cm_resistance_param.update)
            updater.add(link, "length", cm_resistance_param.update)


class approx_chezy_manning_headloss_constraint(Definition):  # noqa: D101
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
        if not hasattr(m, "approx_hazen_williams_headloss"):
            m.approx_chezy_manning_headloss = aml.ConstraintDict()

        if index_over is None:
            index_over = wn.pipe_name_list

        for link_name in index_over:
            if link_name in m.approx_chezy_manning_headloss:
                del m.approx_chezy_manning_headloss[link_name]

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
                k = m.cm_resistance[link_name]

                con = aml.Constraint(expr=-k * f**m.cm_exp + start_h - end_h)

            m.approx_chezy_manning_headloss[link_name] = con

            updater.add(link, "status", approx_chezy_manning_headloss_constraint.update)
            updater.add(
                link, "_is_isolated", approx_chezy_manning_headloss_constraint.update
            )


def get_chezy_manning_qubops_matrix(
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

        # link info
        link = wn.get_link(link_name)

        # get the start/end node info
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
        k = m.cm_resistance[link_name]
        sign_index = flow_index_mapping[m.flow[link_name].name]["sign"]
        flow_index = flow_index_mapping[m.flow[link_name].name]["absolute_value"]
        P3[ieq, sign_index, flow_index, flow_index] -= k.value

    return (P0, P1, P2, P3)


def get_pipe_design_chezy_manning_qubops_matrix(
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
    P0, P1, P2, P3, P4 = matrices
    num_continuous_var = 2 * len(m.flow) + len(m.head)

    for ieq0, link_name in enumerate(wn.pipe_name_list):

        # index of the pipe equation
        ieq = ieq0 + len(wn.junction_name_list)

        # link info
        link = wn.get_link(link_name)

        # get start/end node info
        start_node_name = link.start_node_name
        end_node_name = link.end_node_name
        start_node = wn.get_node(start_node_name)
        end_node = wn.get_node(end_node_name)

        # linear term (start head value) of the headloss approximation
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

        # non linear term (resistance sign flow^2) of the headloss approximation
        sign_index = flow_index_mapping[m.flow[link_name].name]["sign"]
        flow_index = flow_index_mapping[m.flow[link_name].name]["absolute_value"]
        for pipe_coefs, pipe_idx in zip(
            m.pipe_coefficients[link_name].value,
            m.pipe_coefficients_indices[link_name].value,
        ):
            P4[
                ieq, sign_index, flow_index, flow_index, pipe_idx + num_continuous_var
            ] = -pipe_coefs

    return (P0, P1, P2, P3, P4)
