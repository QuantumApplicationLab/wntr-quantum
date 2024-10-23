from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from dimod import as_samples
from tqdm import tqdm


def generate_random_valid_sample(qubo):
    """Geenrate a random sample that respects quadratization.

    Args:
        qubo (_type_): _description_
    """
    sample = {}
    for iv, v in enumerate(sorted(qubo.qubo_dict.variables)):
        sample[v] = np.random.randint(2)

    for v in qubo.mapped_variables[:7]:
        sample[v] = 1
    sample[qubo.mapped_variables[7]] = 0

    for v, _ in sample.items():
        if v not in qubo.mapped_variables:
            var_tmp = v.split("*")
            itmp = 0
            for vtmp in var_tmp:
                if itmp == 0:
                    new_val = sample[vtmp]
                    itmp = 1
                else:
                    new_val *= sample[vtmp]

            sample[v] = new_val
    return sample


def modify_solution_sample(net, solution, modify=["signs", "flows", "heads"]):
    """_summary_

    Args:
        qubo (_type_): _description_
        solution (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    def flatten_list(lst):
        out = []
        for elmt in lst:
            if not isinstance(elmt, list):
                out += [elmt]
            else:
                out += elmt
        return out

    from copy import deepcopy

    for m in modify:
        if m not in ["signs", "flows", "heads"]:
            raise ValueError("modify %s not recognized" % m)

    mod_bin_rep_sol = deepcopy(solution)
    num_pipes = net.wn.num_pipes
    num_heads = net.wn.num_junctions

    # modsify sign
    if "signs" in modify:
        for i in range(num_pipes):
            mod_bin_rep_sol[i] = np.random.randint(2)

    # modify flow value
    if "flows" in modify:
        for i in range(num_pipes, 2 * num_pipes):
            mod_bin_rep_sol[i] = list(
                np.random.randint(2, size=net.flow_encoding.nqbit)
            )

    # modify head values
    if "heads" in modify:
        for i in range(2 * num_pipes, 2 * num_pipes + num_heads):
            mod_bin_rep_sol[i] = list(
                np.random.randint(2, size=net.head_encoding.nqbit)
            )

    x = net.qubo.extend_binary_representation(flatten_list(mod_bin_rep_sol))
    return x


@dataclass
class SimulatedAnnealingResults:
    """Result of the simulated anneling."""

    res: list
    energies: list
    trajectory: list


class SimulatedAnnealing:  # noqa: D101

    def __init__(self):  # noqa: D107
        self.properties = {}

    def optimize_value(self, variables=["sign", "flow", "pressure"]):
        """_summary_.

        Args:
            variables (list, optional): _description_. Defaults to ['sign', 'flow', 'pressure'].
        """

    def sample(
        self,
        bqm,
        num_sweeps=100,
        Temp=[1e5, 1e-3],
        Tschedule=None,
        x0=None,
        take_step=None,
        save_traj=False,
    ):
        """Sample the problem.

        Args:
            bqm (_type_): _description_
            num_sweeps (int, optional): _description_. Defaults to 100.
            Temp (list, optional): _description_. Defaults to [1e5, 1e-3].
            Tschedule (list, optional): The temperature schedule
            x0 (_type_, optional): _description_. Defaults to None.
            take_step (_type_, optional): _description_. Defaults to None.
            save_traj (bool, optional): save the trajectory. Defaults to False
        """

        def bqm_energy(x, var_names):
            """Compute the energy of a given binary array.

            Args:
                x (_type_): _description_
                var_names (list): list of var names
            """
            return bqm.energies(as_samples((x, var_names)))

        # check that take_step is callable
        if not callable(take_step):
            raise ValueError("take_step must be callable")

        # define th variable names
        var_names = sorted(bqm.variables)

        # define the initial state
        if x0 is None:
            x = np.random.randint(2, size=bqm.num_variables)
        else:
            x = x0

        # define the energy range
        if Tschedule is None:
            Tschedule = np.linspace(Temp[0], Temp[1], num_sweeps)

        # initialize the energy
        energies = []
        energies.append(bqm_energy(x, var_names))

        # init the traj
        trajectory = None
        if save_traj:
            trajectory = []
            trajectory.append(x)

        # step scheduling
        # step_schedule = (
        #     Tschedule / ((Tschedule[0] - Tschedule[-1]) / (take_step.step_size - 1)) + 1
        # )

        # loop over the temp schedule
        for T in tqdm(Tschedule):

            # original point
            x_ori = deepcopy(x)
            e_ori = bqm_energy(x, var_names)

            # new point
            # take_step.step_size = int(s)
            x_new = take_step(x)
            e_new = bqm_energy(x, var_names)

            # accept/reject
            if e_new < e_ori:
                x = x_new
                energies.append(bqm_energy(x, var_names))
                if save_traj:
                    trajectory.append(x)
            else:
                if T != 0:
                    p = np.exp(-(e_new - e_ori) / T)
                else:
                    p = 0.0
                if np.random.rand() < p:
                    x = x_new
                    energies.append(bqm_energy(x, var_names))
                    if save_traj:
                        trajectory.append(x)
                else:
                    x = x_ori
                    energies.append(bqm_energy(x, var_names))
                    if save_traj:
                        trajectory.append(x)

        return SimulatedAnnealingResults(x, energies, trajectory)
