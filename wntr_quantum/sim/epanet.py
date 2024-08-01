"""The EPANET simulator."""

import logging
import os
import pickle
import sys
from dataclasses import dataclass
import numpy as np
import wntr.epanet.io
from quantum_newton_raphson.base_solver import BaseSolver
from quantum_newton_raphson.base_solver import ValidInputFormat
from wntr.network.io import write_inpfile
from wntr.sim.epanet import EpanetSimulator
from .results import QuantumSimulationResults

logger = logging.getLogger(__name__)

try:
    import wntr_quantum.epanet.toolkit
except ImportError as e:
    print("{}".format(e))
    logger.critical("%s", e)
    raise ImportError(
        "Error importing epanet toolkit while running epanet simulator. "
        "Make sure libepanet is installed and added to path."
    )


def load_epanet_matrix():
    """Load the data contained in an epanet temp file."""
    epanet_path = os.environ["EPANET_QUANTUM"]
    epanet_tmp = os.environ["EPANET_TMP"]
    util_path = os.path.join(epanet_path, "src/py/")
    # the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
    sys.path.append(util_path)
    from quantum_linsolve import load_json_data

    return load_json_data(os.path.join(epanet_tmp, "smat.json"))


@dataclass
class CholeskyResult:
    """Results of the Sparse LU solver."""

    solution: np.ndarray


class CholeskySolver(BaseSolver):
    """Solve the linear sysem using SPLU.

    Args:
        BaseSolver (class): base class
    """

    def __call__(self, A: ValidInputFormat, b: ValidInputFormat) -> CholeskyResult:
        """Solve the linear system using Cholseky.

        Args:
            A (ValidInputFormat): input matrix
            b (ValidInputFormat): input rhs

        Returns:
            SPLUResult: object containing all the results of the solver
        """
        A = A.todense()
        # print(A)
        # print(b)
        L = np.linalg.cholesky(A)
        # print(L @ L.T)
        y = np.linalg.solve(L, b)
        x = np.linalg.solve(L.T, y)
        return CholeskyResult(x)


class QuantumEpanetSimulator(EpanetSimulator):
    """Fast EPANET simulator class.

    Use the EPANET DLL to run an INP file as-is, and read the results from the
    binary output file. Multiple water quality simulations are still possible
    using the WQ keyword in the run_sim function. Hydraulics will be stored and
    saved to a file. This file will not be deleted by default, nor will any
    binary files be deleted.

    The reason this is considered a "fast" simulator is due to the fact that there
    is no looping within Python. The "ENsolveH" and "ENsolveQ" toolkit
    functions are used instead.


    .. note::

        WNTR now includes access to both the EPANET 2.0.12 and EPANET 2.2 toolkit libraries.
        By default, version 2.2 will be used.


    Parameters
    ----------
    wn : WaterNetworkModel
        Water network model
    reader : wntr.epanet.io.BinFile derived object
        Defaults to None, which will create a new wntr.epanet.io.BinFile object with
        the results_types specified as an init option. Otherwise, a fully
    result_types : dict
        Defaults to None, or all results. Otherwise, is a keyword dictionary to pass to
        the reader to specify what results should be saved.


    .. seealso::

        wntr.epanet.io.BinFile

    """

    def __init__(  # noqa: D107
        self, wn, reader=None, result_types=None, linear_solver=CholeskySolver()
    ):
        EpanetSimulator.__init__(self, wn)
        self.reader = reader
        self.prep_time_before_main_loop = 0.0
        if self.reader is None:
            self.reader = wntr.epanet.io.BinFile(result_types=result_types)
        self.linear_solver = linear_solver
        if "EPANET_TMP" not in os.environ:
            raise ValueError(
                "Please define the environment variable EPANET_TMP with \
                             the path to a folder where you have write access"
            )
        self.epanet_shared = os.environ["EPANET_TMP"]
        self.clean_up_epanet_tmp()

        # check if the en var of the source code exists and if the python file is ther
        if "EPANET_QUANTUM" not in os.environ:
            raise ValueError(
                "Please define the environment variable EPANET_QUANTUM with \
                the path to the root of the EPANET_QUANTUM directory"
            )
        else:
            py_path = os.environ["EPANET_QUANTUM"]
            py_path = os.path.join(py_path, "src")
            py_path = os.path.join(py_path, "py")
            py_path = os.path.join(py_path, "quantum_linsolve.py")
            if not os.path.isfile(py_path):
                raise FileNotFoundError("%s file not found" % py_path)

        # dump the solver
        with open(os.path.join(self.epanet_shared, "solver.pckl"), "wb") as fb:
            pickle.dump(linear_solver, fb)

    def clean_up_epanet_tmp(self):
        """Clean up EPANET_TMP if it is not empty."""
        if os.listdir(self.epanet_shared):
            files = os.listdir(self.epanet_shared)
            for file in files:
                file_path = os.path.join(self.epanet_shared, file)
                os.remove(file_path)
        else:
            logger.info(f"Empty EPANET_TMP: {self.epanet_shared}")

    def run_sim(
        self,
        file_prefix="temp",
        save_hyd=False,
        use_hyd=False,
        hydfile=None,
        version=2.2,
        convergence_error=False,
        linear_solver=CholeskySolver(),
    ):
        """Run the EPANET simulator.

        Runs the EPANET simulator through the compiled toolkit DLL. Can use/save hydraulics
        to allow for separate WQ runs.

        .. note::

            By default, WNTR now uses the EPANET 2.2 toolkit as the engine for the EpanetSimulator.
            To force usage of the older EPANET 2.0 toolkit, use the ``version`` command line option.
            Note that if the demand_model option is set to PDD, then a warning will be issued, as
            EPANET 2.0 does not support such analysis.


        Parameters
        ----------
        file_prefix : str
            Default prefix is "temp". All files (.inp, .bin/.out, .hyd, .rpt) use this prefix
        use_hyd : bool
            Will load hydraulics from ``file_prefix + '.hyd'`` or from file specified in `hydfile_name`
        save_hyd : bool
            Will save hydraulics to ``file_prefix + '.hyd'`` or to file specified in `hydfile_name`
        hydfile : str
            Optionally specify a filename for the hydraulics file other than the `file_prefix`
        version : float, {2.0, **2.2**}
            Optionally change the version of the EPANET toolkit libraries. Valid choices are
            either 2.2 (the default if no argument provided) or 2.0.
        convergence_error: bool (optional)
            If convergence_error is True, an error will be raised if the
            simulation does not converge. If convergence_error is False, partial results are returned,
            a warning will be issued, and results.error_code will be set to 0
            if the simulation does not converge.  Default = False.
        linear_solver: BaseLinearSolver (optional)
            The linear solver we want to use for the NR step.
        """
        if isinstance(version, str):
            version = float(version)
        inpfile = file_prefix + ".inp"

        write_inpfile(
            self._wn,
            inpfile,
            units=self._wn.options.hydraulic.inpfile_units,
            version=version,
        )
        # self._wn.write_inpfile(
        #     inpfile, units=self._wn.options.hydraulic.inpfile_units, version=version
        # )

        with open(os.path.join(self.epanet_shared, "solver.pckl"), "wb") as fb:
            pickle.dump(linear_solver, fb)

        enData = wntr_quantum.epanet.toolkit.ENepanet_quantum(version=version)
        rptfile = file_prefix + ".rpt"
        outfile = file_prefix + ".bin"

        if hydfile is None:
            hydfile = file_prefix + ".hyd"
        enData.ENopen(inpfile, rptfile, outfile)
        if use_hyd:
            enData.ENusehydfile(hydfile)
            logger.debug("Loaded hydraulics")
        else:
            enData.ENsolveH()
            logger.debug("Solved hydraulics")
        if save_hyd:
            enData.ENsavehydfile(hydfile)
            logger.debug("Saved hydraulics")
        enData.ENsolveQ()
        logger.debug("Solved quality")
        enData.ENreport()
        logger.debug("Ran quality")
        enData.ENclose()
        logger.debug("Completed run")
        # os.sys.stderr.write('Finished Closing\n')

        results = self.reader.read(
            outfile, convergence_error, self._wn.options.hydraulic.headloss == "D-W"
        )

        # convert instance of `SimulationResults` into `QuantumSimulationResults`
        results = QuantumSimulationResults.from_simulation_results(results)

        # if relevant, save linear solver intermediate results
        sol_info = os.path.join(self.epanet_shared, "sol_info.pckl")
        if os.path.isfile(sol_info):
            with open(sol_info, "rb") as f:
                linear_solver_results = pickle.load(f)
            results.linear_solver_results = linear_solver_results

        return results
