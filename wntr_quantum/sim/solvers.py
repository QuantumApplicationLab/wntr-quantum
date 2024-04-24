import numpy as np
import scipy.sparse as sp
import warnings
import logging
import time
from wntr.sim.solvers import NewtonSolver, SolverStatus
from quantum_newton_raphson.splu_solver import SPLU_SOLVER 
from quantum_newton_raphson.result import SPLUResult, VQLSResult, QUBOResult 

warnings.filterwarnings(
    "error", "Matrix is exactly singular", sp.linalg.MatrixRankWarning
)
np.set_printoptions(precision=3, threshold=10000, linewidth=300)

logger = logging.getLogger(__name__)

def get_solution_vector(result: SPLU_SOLVER | VQLSResult | QUBOResult) -> np.ndarray:
    """Get the soluion vector from the result dataclass

    Args:
        result (SPLU_SOLVER | VQLSResult | QUBOResult): linear solver dataclass result

    Returns:
        np.ndarray: solution vector
    """
    return result.solution

class QuantumNewtonSolver(NewtonSolver):
    """
    Quantum Newton Solver class.

    Attributes
    ----------
    log_progress: bool
        If True, the infinity norm of the constraint violation will be logged each iteration
    log_level: int
        The level for logging the infinity norm of the constraint violation
    time_limit: float
        If the wallclock time exceeds time_limit, the newton solver will exit with an error status
    maxiter: int
        If the number of iterations exceeds maxiter, the newton solver will exit with an error status
    tol: float
        The convergence tolerance. If the infinity norm of the constraint violation drops below tol,
        the newton solver will exit with a converged status.
    rho: float
        During the line search, rho is used to reduce the stepsize. It should be strictly between 0 and 1.
    bt_maxiter: int
        The maximum number of line search iterations for each outer iteration
    bt: bool
        If False, a line search will not be used.
    bt_start_iter: int
        A line search will not be used for any iteration prior to bt_start_iter
    """

    def __init__(self, linear_solver=SPLU_SOLVER(), options=None):
        super().__init__(options)
        self._linear_solver = linear_solver


    def solve(self, model, ostream=None):
        """

        Parameters
        ----------
        model: wntr.aml.Model

        Returns
        -------
        status: SolverStatus
        message: str
        iter_count: int
        """
        t0 = time.time()

        x = model.get_x()
        if len(x) == 0:
            return (
                SolverStatus.converged,
                "No variables or constraints",
                0,
            )

        use_r_ = False

        # MAIN NEWTON LOOP
        for outer_iter in range(self.maxiter):
            if time.time() - t0 >= self.time_limit:
                return (
                    SolverStatus.error,
                    "Time limit exceeded",
                    outer_iter,
                )

            if use_r_:
                r = r_
                r_norm = new_norm
            else:
                r = model.evaluate_residuals()
                r_norm = np.max(abs(r))

            if self.log_progress or ostream is not None:
                if outer_iter < self.bt_start_iter:
                    msg = f"iter: {outer_iter:<4d} norm: {r_norm:<10.2e} time: {time.time() - t0:<8.4f}"
                    if self.log_progress:
                        logger.log(self.log_level, msg)
                    if ostream is not None:
                        ostream.write(msg + "\n")

            if r_norm < self.tol:
                return (
                    SolverStatus.converged,
                    "Solved Successfully",
                    outer_iter,
                )

            J = model.evaluate_jacobian(x=None)

            # Call Linear solver
            try:
                d = -get_solution_vector(self._linear_solver(J, r))
                # d = -sp.linalg.spsolve(J, r, permc_spec="COLAMD", use_umfpack=False)
            except sp.linalg.MatrixRankWarning:
                return (
                    SolverStatus.error,
                    "Jacobian is singular at iteration " + str(outer_iter),
                    outer_iter,
                )

            # Backtracking
            alpha = 1.0
            if self.bt and outer_iter >= self.bt_start_iter:
                use_r_ = True
                for iter_bt in range(self.bt_maxiter):
                    x_ = x + alpha * d
                    model.load_var_values_from_x(x_)
                    r_ = model.evaluate_residuals()
                    new_norm = np.max(abs(r_))
                    if new_norm < (1.0 - 0.0001 * alpha) * r_norm:
                        x = x_
                        break
                    else:
                        alpha = alpha * self.rho

                if iter_bt + 1 >= self.bt_maxiter:
                    return (
                        SolverStatus.error,
                        "Line search failed at iteration " + str(outer_iter),
                        outer_iter,
                    )
                if self.log_progress or ostream is not None:
                    msg = f"iter: {outer_iter:<4d} norm: {new_norm:<10.2e} alpha: {alpha:<10.2e} time: {time.time() - t0:<8.4f}"
                    if self.log_progress:
                        logger.log(self.log_level, msg)
                    if ostream is not None:
                        ostream.write(msg + "\n")
            else:
                x += d
                model.load_var_values_from_x(x)

        return (
            SolverStatus.error,
            "Reached maximum number of iterations: " + str(outer_iter),
            outer_iter,
        )
