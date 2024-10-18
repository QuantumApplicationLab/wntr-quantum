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


@dataclass
class SimulatedAnnealingResults:
    """Result of the simulated anneling."""

    res: list
    energies: list
    trajectory: list


class SimulatedAnnealing:  # noqa: D101

    def __init__(self):  # noqa: D107
        self.properties = {}

    def sample(
        self,
        bqm,
        Tschedule,
        num_traj=10,
        x0=None,
        take_step=None,
        save_traj=False,
    ):
        """Sample the problem.

        Args:
            bqm (_type_): _description_
            Tschedule (list): The temperature schedule
            x0 (_type_, optional): _description_. Defaults to None.
            num_traj(int, optional): number of parallel traj. Default to None
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
            x = np.random.randint(2, size=(num_traj, bqm.num_variables)).tolist()
        else:
            x = x0

        # initialize the energy
        energies = []
        energies.append(bqm_energy(x, var_names))

        # init the traj
        trajectory = None
        if save_traj:
            trajectory = []
            trajectory.append(x)

        # step scheduling
        step_schedule = (
            Tschedule / ((Tschedule[0] - Tschedule[-1]) / (take_step.step_size - 1)) + 1
        )

        # loop over the temp schedule
        for s, T in tqdm(zip(step_schedule, Tschedule)):

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
                p = np.exp(-(e_new - e_ori) / T)
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
