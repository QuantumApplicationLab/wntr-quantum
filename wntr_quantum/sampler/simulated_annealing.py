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


class SimulatedAnnealing:  # noqa: D101

    def __init__(self):  # noqa: D107
        self.properties = {}

    def sample(
        self,
        bqm,
        num_sweeps=100,
        Temp=[1e5, 1e-3],
        Tschedule=None,
        x0=None,
        take_step=None,
    ):
        """Sample the problem.

        Args:
            bqm (_type_): _description_
            num_sweeps (int, optional): _description_. Defaults to 100.
            Temp (list, optional): _description_. Defaults to [1e5, 1e-3].
            Tschedule (list, optional): The temperature schedule
            x0 (_type_, optional): _description_. Defaults to None.
            take_step (_type_, optional): _description_. Defaults to None.
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

        # loop over the temp schedule
        for T in tqdm(Tschedule):

            # original point
            x_ori = deepcopy(x)
            e_ori = bqm_energy(x, var_names)

            # new point
            x_new = take_step(x)
            e_new = bqm_energy(x, var_names)

            # accept/reject
            if e_new < e_ori:
                x = x_new
                energies.append(bqm_energy(x, var_names))
            else:
                p = np.exp(-(e_new - e_ori) / T)
                if np.random.rand() < p:
                    x = x_new
                    energies.append(bqm_energy(x, var_names))
                else:
                    x = x_ori

        return SimulatedAnnealingResults(x, energies)
