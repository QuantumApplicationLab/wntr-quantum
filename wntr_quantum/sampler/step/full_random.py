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
