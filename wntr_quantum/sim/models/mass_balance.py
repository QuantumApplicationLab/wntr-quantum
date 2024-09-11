from wntr.epanet.util import FlowUnits
from wntr.epanet.util import HydParam
from wntr.epanet.util import from_si


def get_mass_balance_qubops_matrix(
    m, wn, matrices, convert_to_us_unit=False
):  # noqa: D417
    """Create the matrices for the mass balance equation.

    Args:
        m (aml.Model): The AML model of the network
        wn (WaternNetwork): th water network object
        matrices (Tuple): The qubops matrices of the network
        convert_to_us_unit (bool, optional): Convert the inut to US units. Defaults to False.

    Returns:
        Tuple: The qubops matrices of the network
    """
    P0, P1, P2 = matrices

    continuous_var_name = [v.name for v in list(m.vars())]
    var_names = continuous_var_name

    index_over = wn.junction_name_list

    for ieq, node_name in enumerate(index_over):

        node = wn.get_node(node_name)
        if not node._is_isolated:
            if convert_to_us_unit:
                P0[ieq, 0] += from_si(
                    FlowUnits.CFS, m.expected_demand[node_name].value, HydParam.Flow
                )
            else:
                P0[ieq, 0] += m.expected_demand[node_name].value

            for link_name in wn.get_links_for_node(node_name, flag="INLET"):
                link_index = var_names.index(m.flow[link_name].name)
                P1[ieq, link_index] -= 1

            for link_name in wn.get_links_for_node(node_name, flag="OUTLET"):
                link_index = var_names.index(m.flow[link_name].name)
                P1[ieq, link_index] += 1

    return P0, P1, P2


def get_mass_balance_constraint_design(m, wn, matrices):  # noqa: D417
    """Adds a mass balance to the model for the specified junctions.

    Parameters
    ----------
    m: wntr.aml.aml.aml.Model
    wn: wntr.network.model.WaterNetworkModel
    updater: ModelUpdater
    index_over: list of str
        list of junction names; default is all junctions in wn
    """
    P0, P1, P2, P3 = matrices

    continuous_var_name = [v.name for v in list(m.vars())]
    discrete_var_name = [v.name for k, v in m.cm_resistance.items()]
    var_names = continuous_var_name + discrete_var_name

    index_over = wn.junction_name_list

    for ieq, node_name in enumerate(index_over):

        node = wn.get_node(node_name)
        if not node._is_isolated:
            P0[ieq, 0] += m.expected_demand[node_name].value

            for link_name in wn.get_links_for_node(node_name, flag="INLET"):
                node_index = var_names.index(m.flow[link_name].name)
                P1[ieq, node_index] -= 1

            for link_name in wn.get_links_for_node(node_name, flag="OUTLET"):
                node_index = var_names.index(m.flow[link_name].name)
                P1[ieq, node_index] += 1

    return P0, P1, P2, P3
