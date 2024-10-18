import numpy as np
from .base_step import BaseStep


class RandomStep(BaseStep):  # noqa: D101

    def __call__(self, x):
        """Call function of the method.

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        nmax = 8 + 8 * 7
        vidx = np.random.choice(self.single_var_index[nmax:])
        x[vidx] = int(not (x[vidx]))
        self.fix_constraint(x, vidx)
        return x


class IncrementalStep(BaseStep):

    def __init__(
        self,
        var_names,
        single_var_names,
        single_var_index,
        step_size=1,
        optimize_values=None,
    ):
        super().__init__(var_names, single_var_names, single_var_index)

        self.value_names = np.unique(
            [self._get_variable_root_name(n) for n in single_var_names]
        )
        self.index_values = {v: [] for v in self.value_names}
        for n, idx in zip(self.single_var_names, self.single_var_index):
            val = self._get_variable_root_name(n)
            self.index_values[val].append(idx)

        self.step_size = step_size
        self.optimize_values = optimize_values
        if self.optimize_values is None:
            self.optimize_values = list(np.arange(len(self.value_names)))

    @staticmethod
    def _get_variable_root_name(var_name):
        """Extract the root name of the variables.

        Args:
            var_name (_type_): _description_
        """
        return "_".join(var_name.split("_")[:2])

    def __call__(self, x):
        """Call function of the method.

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        random_val_name = np.random.choice(self.value_names[self.optimize_values])
        idx = self.index_values[random_val_name]
        data = np.array(x)[idx]
        width = len(data)

        # determine the max val
        max_val = np.ones_like(data)
        max_val = int("".join([str(i) for i in max_val[::-1]]), base=2)

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

        # inject in the x vector
        for ix, nd in zip(idx, new_data):
            x[ix] = nd

        # fix constraints
        for vidx in idx:
            self.fix_constraint(x, vidx)

        return x


class ParallelIncrementalStep(BaseStep):

    def __init__(self, var_names, single_var_names, single_var_index, step_size=1):
        super().__init__(var_names, single_var_names, single_var_index)
        self.step_size = step_size
        self._step = IncrementalStep(
            var_names, single_var_names, single_var_index, step_size=step_size
        )

    def __call__(self, x):
        """Call function of the method.

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        new_x = []
        for xi in x:
            new_x.append(self._step(xi))
        return new_x
