from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm


def generate_random_valid_sample(net):
    """Generate a random sample that respects quadratization.

    Args:
        net (QuboPolynomialSolver): an instance of the QuboPolynomialSolver
    """
    sample = {}
    for iv, v in enumerate(sorted(net.qubo.qubo_dict.variables)):
        sample[v] = np.random.randint(2)

    for v in net.qubo.mapped_variables[:7]:
        sample[v] = 1
    sample[net.qubo.mapped_variables[7]] = 0

    for v, _ in sample.items():
        if v not in net.qubo.mapped_variables:
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


def modify_solution_sample(net, solution, modify=["signs", "flows", "heads"]) -> list:
    """Modiy the solution sample to change values of the signs/flows/heads.

    Args:
        net (QuboPolynomialSolver): an instance of the QuboPolynomialSolver
        solution (list): the sample that encoded the true solution
        modify (list, optional): what to change. Defaults to ["signs", "flows", "heads"].

    Returns:
        List: new sample
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

    # modify sign
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
        self.Tschedule = None
        self.init_sample = None
        self.take_step = None
        self.save_traj = False

    @property
    def Tschedule(self):  # noqa: D102
        return self._Tschedule

    @Tschedule.setter
    def Tschedule(self, tschedule):
        self._Tschedule = tschedule

    @property
    def init_sample(self):  # noqa: D102
        return self._init_sample

    @init_sample.setter
    def init_sample(self, sample):
        self._init_sample = sample

    @property
    def take_step(self):  # noqa: D102
        return self._take_step

    @take_step.setter
    def take_step(self, step):
        self._take_step = step

    def sample(
        self,
        qubo,
        Tschedule=None,
        init_sample=None,
        take_step=None,
        save_traj=False,
        verbose=False,
    ):
        """Sample the problem.

        Args:
            qubo (qubo solver): qubo solver
            Tschedule (list, optional): The temperature schedule
            init_sample (list, optional): initial sample for the optimzation. Defaults to None.
            take_step (callable, optional): Callable to obtain new sample. Defaults to None.
            save_traj (bool, optional): save the trajectory. Defaults to False
            verbose (bool, optional): print stuff
        """

        def bqm_energy(qubo, input, var_names):
            """Computes the energy of the sample.

            Args:
                qubo (qubo_solver): qubo solver
                input (list): sample
                var_names (list): names of the variables


            Returns:
                float: qubo energy
            """
            return qubo.energy_binary_rep(
                np.array(input)[qubo.index_variables].tolist()
            )

        if Tschedule is not None:
            self.Tschedule = Tschedule

        if init_sample is not None:
            self.init_sample = init_sample

        if take_step is not None:
            self.take_step = take_step

        self.save_traj = save_traj

        self.bqm = qubo.qubo_dict

        # check that take_step is callable
        if not callable(self.take_step):
            raise ValueError("take_step must be callable")

        # define the variable names
        self.var_names = sorted(self.bqm.variables)

        # define the initial state
        if self.init_sample is None:
            current_sample = generate_random_valid_sample(self.bqm)
        else:
            current_sample = init_sample

        # init the traj
        trajectory = []
        if self.save_traj:
            trajectory.append(current_sample)

        # initialize the energy
        energies = []
        e_current = bqm_energy(qubo, current_sample, self.var_names)
        energies.append(e_current)

        # loop over the temp schedule
        for T in tqdm(self.Tschedule):

            # new point
            new_sample = self.take_step(deepcopy(current_sample), verbose=verbose)
            e_new = bqm_energy(qubo, new_sample, self.var_names)

            # accept/reject
            if e_new < e_current:
                if verbose:
                    print("E :  %f =>  %f" % (e_current, e_new))
                current_sample = deepcopy(new_sample)
                e_current = e_new

            else:
                if verbose:
                    print("E :  %f =>  %f" % (e_current, e_new))

                p = np.exp((e_current - e_new) / (T + 1e-12))
                eps = np.random.rand()

                if eps < p:
                    current_sample = deepcopy(new_sample)
                    e_current = e_new

                else:
                    if verbose:
                        print("rejected")
                    pass

            if self.save_traj:
                trajectory.append(current_sample)
            energies.append(e_current)

            if verbose:
                print("-----------------")
        return SimulatedAnnealingResults(current_sample, energies, trajectory)
