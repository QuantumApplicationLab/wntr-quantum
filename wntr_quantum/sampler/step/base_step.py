import numpy as np


class BaseStep:  # noqa: D101
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
        raise NotImplementedError("Implement a __call__ method")
