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
    m.cm_k = 4.66
    m.cm_diameter_exp = -5.33


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
            value = (
                m.cm_k
                * link.roughness ** (m.cm_exp)
                * link.diameter ** (m.cm_diameter_exp)
                * link.length
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
                eps = 1e-5  # Need to provide an options for this
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
                minor_k = m.minor_loss[link_name]

                con = aml.Constraint(expr=-k * f**m.cm_exp + start_h - end_h)

            m.approx_chezy_manning_headloss[link_name] = con

            updater.add(link, "status", approx_chezy_manning_headloss_constraint.update)
            updater.add(
                link, "_is_isolated", approx_chezy_manning_headloss_constraint.update
            )
