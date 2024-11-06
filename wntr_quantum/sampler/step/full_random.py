import numpy as np
from .base_step import BaseStep


class RandomStep(BaseStep):  # noqa: D101

    def __call__(self, x, verbose=False):
        """Call function of the method.

        Args:
            x (list): initial sample
            verbose (bool): print stuff

        Returns:
            list: proposed sample
        """
        random_val_name = np.random.choice(self.value_names[self.optimize_values])
        idx = self.index_values[random_val_name]
        vidx = np.random.choice(idx)
        x[vidx] = int(not (x[vidx]))
        self.fix_constraint(x, vidx)
        return x


class IncrementalStep(BaseStep):  # noqa: D101

    def __call__(self, x, verbose=False):
        """Call function of the method.

        Args:
            x (list): initial sample
            verbose (bool): print stuff

        Returns:
            list: proposed sample
        """
        num_var_changed = np.random.randint(len(self.optimize_values))
        random_val_name_list = np.random.choice(
            self.value_names[self.optimize_values], size=num_var_changed
        )

        for random_val_name in random_val_name_list:
            idx = self.index_values[random_val_name]
            data = np.array(x)[idx]
            width = len(data)

            # determine the max val
            max_val = int("1" * width, base=2)

            # check if we reach min/max val
            max_val_check = data.prod() == 1
            min_val_check = data.sum() == 0

            # convert to int value
            val = int("".join([str(i) for i in data[::-1]]), base=2)

            # determine sign of the displacement
            if min_val_check:
                sign = 1
            elif max_val_check:
                sign = -1
            else:
                sign = 2 * np.random.randint(2) - 1

            # new value
            if self.step_size <= 1:
                delta = 1
            else:
                delta = np.random.randint(self.step_size)
            new_val = val + sign * delta
            if new_val < 0:
                new_val = 0
            if new_val > max_val:
                new_val = max_val
            new_val = np.binary_repr(new_val, width=width)

            # convert back to binary repr
            new_data = np.array([int(i) for i in new_val])[::-1]
            if verbose:
                print(random_val_name, data, "=>", new_data)

            # inject in the x vector
            for ix, nd in zip(idx, new_data):
                x[ix] = nd

            # fix constraints
            for vidx in idx:
                self.fix_constraint(x, vidx)

        return x


class SwitchIncrementalStep(BaseStep):  # noqa: D101

    def __init__(  # noqa: D417
        self,
        var_names,
        single_var_names,
        single_var_index,
        switch_variable_index,
        step_size=1,
        optimize_values=None,
    ):
        """Propose a new solution vector.

        Args:
            var_names (list): names of the variables in the problem
            single_var_names (_type_): list of the single variables names e.g. x_001_002
            single_var_index (_type_): index of the single variables
            switch_variable_index (list): index of the variables we are switching over
            step_size (int, optional): size of the steps
            optimize_values (list, optional): index of the values to optimize
        """
        super().__init__(
            var_names, single_var_names, single_var_index, step_size, optimize_values
        )
        self.switch_variable_index = switch_variable_index

    def __call__(self, x, verbose=False):
        """Call function of the method.

        Args:
            x (list): initial sample
            verbose (bool): print stuff

        Returns:
            list: proposed sample
        """
        num_var_changed = np.random.randint(len(self.optimize_values))
        random_val_name_list = np.random.choice(
            self.value_names[self.optimize_values], size=num_var_changed
        )

        for random_val_name in random_val_name_list:
            idx = self.index_values[random_val_name]
            data = np.array(x)[idx]
            width = len(data)

            # determine the max val
            max_val = int("1" * width, base=2)

            # check if we reach min/max val
            max_val_check = data.prod() == 1
            min_val_check = data.sum() == 0

            # convert to int value
            val = int("".join([str(i) for i in data[::-1]]), base=2)

            # determine sign of the displacement
            if min_val_check:
                sign = 1
            elif max_val_check:
                sign = -1
            else:
                sign = 2 * np.random.randint(2) - 1

            # new value
            if self.step_size <= 1:
                delta = 1
            else:
                delta = np.random.randint(self.step_size)
            new_val = val + sign * delta
            if new_val < 0:
                new_val = 0
            if new_val > max_val:
                new_val = max_val
            new_val = np.binary_repr(new_val, width=width)

            # convert back to binary repr
            new_data = np.array([int(i) for i in new_val])[::-1]
            if verbose:
                print(random_val_name, data, "=>", new_data)

            # inject in the x vector
            for ix, nd in zip(idx, new_data):
                x[ix] = nd

            # fix constraints
            for vidx in idx:
                self.fix_constraint(x, vidx)

        return x
