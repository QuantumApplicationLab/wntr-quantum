import wntr
from wntr.network import LinkStatus
from wntr.sim import aml
from wntr.sim.models.utils import Definition


def chezy_manning_constants(m):
    """Add cehzy manning constants to the model.

    Args:
        m (_type_): _description_
    """
    m.cm_exp = 2
    m.cm_minor_exp = 2
    m.cm_diameter_exp = -5.33
    m.cm_k = 21.000  # 4.66 * (3.28) ** m.cm_diameter_exp

    # m.cm_exp = 2
    # m.cm_minor_exp = 2
    # m.cm_diameter_exp = -4.8
    # m.cm_k = 10.67  # 4.66 * (3.28) ** m.cm_diameter_exp

    # m.cm_exp = 1
    # m.cm_minor_exp = 1
    # m.cm_k = 1
    # m.cm_diameter_exp = -1


def cm_resistance_prefactor(k, roughness, exp, diameter, diameter_exp):
    """_summary_.

    Args:
        k (_type_): _description_
        roughness (_type_): _description_
        exp (_type_): _description_
        diameter (_type_): _description_
        diameter_exp (_type_): _description_
    """
    return k * roughness ** (exp) * diameter ** (diameter_exp)


def cm_resistance_value(k, roughness, exp, diameter, diameter_exp, length):
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
    return cm_resistance_prefactor(k, roughness, exp, diameter, diameter_exp) * length


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
        if not hasattr(m, "hw_resistance"):
            m.cm_resistance = aml.ParamDict()

        if index_over is None:
            index_over = wn.pipe_name_list

        for link_name in index_over:
            link = wn.get_link(link_name)
            value = cm_resistance_value(
                m.cm_k,
                link.roughness,
                m.cm_exp,
                link.diameter,
                m.cm_diameter_exp,
                link.length,
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


def get_chezy_manning_matrix(m, wn, matrices):  # noqa: D417
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
    discrete_var_name = [v.name for k, v in m.cm_resistance.items()]

    var_names = continuous_var_name + discrete_var_name

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
            P0[ieq, 0] += start_h.value

        if isinstance(end_node, wntr.network.Junction):
            end_h = m.head[end_node_name]
            end_node_index = var_names.index(end_h.name)
            P1[ieq, end_node_index] = -1
        else:
            end_h = m.source_head[end_node_name]
            P0[ieq, 0] -= end_h.value

        k = m.cm_resistance[link_name]

        P2[ieq, flow_index, flow_index] = -k.value

    return (P0, P1, P2)


def get_chezy_manning_matrix_design(m, wn, matrices):  # noqa: D417
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
    discrete_var_name = [v.name for k, v in m.cm_resistance.items()]

    var_names = continuous_var_name + discrete_var_name

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
            P0[ieq, 0] += start_h.value

        if isinstance(end_node, wntr.network.Junction):
            end_h = m.head[end_node_name]
            end_node_index = var_names.index(end_h.name)
            P1[ieq, end_node_index] = -1
        else:
            end_h = m.source_head[end_node_name]
            P0[ieq, 0] -= end_h.value

        k = m.cm_resistance[link_name]
        cm_res_index = var_names.index(k.name)

        P3[ieq, flow_index, flow_index, cm_res_index] = -link.length

    return (P0, P1, P2, P3)
