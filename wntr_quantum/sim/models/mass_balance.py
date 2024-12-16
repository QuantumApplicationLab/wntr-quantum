from wntr.epanet.util import FlowUnits
from wntr.epanet.util import HydParam
from wntr.epanet.util import from_si


def get_mass_balance_qubops_matrix(
    m, wn, matrices, flow_index_mapping, convert_to_us_unit=False
):  # noqa: D417
    """Create the matrices for the mass balance equation.

    Args:
        m (aml.Model): The AML model of the network
        wn (WaternNetwork): th water network object
        matrices (Tuple): The qubops matrices of the network
        flow_index_mapping (Dict): A dict to map the flow model variables to the qubops matrices
        convert_to_us_unit (bool, optional): Convert the inut to US units. Defaults to False.

    Returns:
        Tuple: The qubops matrices of the network
    """
    P0, P1, P2, P3 = matrices
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
                sign_idx = flow_index_mapping[m.flow[link_name].name]["sign"]
                flow_idx = flow_index_mapping[m.flow[link_name].name]["absolute_value"]
                P2[ieq, sign_idx, flow_idx] -= 1

            for link_name in wn.get_links_for_node(node_name, flag="OUTLET"):
                sign_idx = flow_index_mapping[m.flow[link_name].name]["sign"]
                flow_idx = flow_index_mapping[m.flow[link_name].name]["absolute_value"]
                P2[ieq, sign_idx, flow_idx] += 1

    return P0, P1, P2, P3
