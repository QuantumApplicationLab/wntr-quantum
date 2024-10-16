from copy import deepcopy
import numpy as np
from dimod import as_samples
from tqdm import tqdm
from dataclasses import dataclass


def generate_random_valid_sample(qubo):
    """Geenrate a random sample that respects quadratization.

    Args:
        qubo (_type_): _description_
    """
    sample = {}
    for iv, v in enumerate(sorted(qubo.qubo_dict.variables)):
        sample[v] = np.random.randint(2)

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


class ProposalStep:  # noqa: D101
    def __init__(self, var_names, single_var_names, single_var_index):
        """Propose a new solution vector.

        Args:
            var_names (_type_): _description_
            single_var_names (_type_): _description_
            single_var_index (_type_): _description_
        """
        self.var_names = var_names
        self.single_var_names = single_var_names
        self.single_var_index = single_var_index
        self.num_single_var = len(self.single_var_names)
        self.high_order_terms_mapping = self.define_mapping()

    def define_mapping(self):
        """Define the mapping of the higher order terms.

        Returns:
            _type_: _description_
        """
        high_order_terms_mapping = []

        # loop over all the variables
        for iv, v in enumerate(self.var_names):

            # if we have a cmomposite variables e.g. x_001 * x_002 we ignore it
            if v not in self.single_var_names:
                high_order_terms_mapping.append(None)

            # if the variables is a unique one e.g. x_011
            else:
                high_order_terms_mapping.append({})
                # we loop over all the variables
                for iiv, vv in enumerate(self.var_names):
                    if v != vv:
                        if v in vv:

                            var_tmp = vv.split("*")
                            idx_terms = []
                            for vtmp in var_tmp:
                                idx = self.single_var_index[
                                    self.single_var_names.index(vtmp)
                                ]
                                idx_terms.append(idx)
                            high_order_terms_mapping[-1][iiv] = idx_terms

        return high_order_terms_mapping

    def fix_constraint(self, x, idx):
        """Ensure that the solution vectors respect quadratization.

        Args:
            x (_type_): _description_
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        fix_var = self.high_order_terms_mapping[idx]
        for idx_fix, idx_prods in fix_var.items():
            x[idx_fix] = np.array([x[i] for i in idx_prods]).prod()
        return x

    def verify_quadratic_constraints(self, data):
        """Check if quadratic constraints are respected or not.

        Args:
            data (_type_): _description_
        """
        for v, d in zip(self.var_names, data):
            if v not in self.single_var_names:
                var_tmp = v.split("*")
                itmp = 0
                for vtmp in var_tmp:
                    idx = self.single_var_index[self.single_var_names.index(vtmp)]
                    if itmp == 0:
                        dcomposite = data[idx]
                        itmp = 1
                    else:
                        dcomposite *= data[idx]
                if d != dcomposite:
                    print("Error in the quadratic contraints")
                    print("%s = %d" % (v, d))
                    for vtmp in var_tmp:
                        idx = self.single_var_index[self.single_var_names.index(vtmp)]
                        print("%s = %d" % (vtmp, data[idx]))

    def __call__(self, x):
        """Call function of the method.

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        vidx = np.random.choice(self.single_var_index)
        x[vidx] = int(not (x[vidx]))
        self.fix_constraint(x, vidx)
        return x


@dataclass
class SimulatedAnnealingResults:
    """Result of the simulated nnelaings"""

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
