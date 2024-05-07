import logging
import time
import warnings
import numpy as np
import scipy.sparse as sp
from wntr.sim.solvers import NewtonSolver
from wntr.sim.solvers import SolverStatus

warnings.filterwarnings(
    "error", "Matrix is exactly singular", sp.linalg.MatrixRankWarning
)
np.set_printoptions(precision=3, threshold=10000, linewidth=300)

logger = logging.getLogger(__name__)


def get_solution_vector(result) -> np.ndarray:
    """Get the soluion vector from the result dataclass.

    Args:
        result (SPLU_SOLVER | VQLSResult | QUBOResult): linear solver dataclass result

    Returns:
        np.ndarray: solution vector
    """
    return result.solution


class QuantumNewtonSolver(NewtonSolver):
    """Quantum Newton Solver class.

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

    def __init__(self, linear_solver, options=None):
        """Init the solver.

        Args:
            linear_solver (SolverBase, optional): Linear solver for the NR step. Defaults to SPLU_SOLVER().
            options (_type_, optional): options for the NR solver. Defaults to None.
        """
        super().__init__(options)

        if "LOG_PROGRESS" not in self._options:
            self.log_progress = False
        else:
            self.log_progress = self._options["LOG_PROGRESS"]

        if "LOG_LEVEL" not in self._options:
            self.log_level = logging.DEBUG
        else:
            self.log_level = self._options["LOG_LEVEL"]

        if "TIME_LIMIT" not in self._options:
            self.time_limit = 3600
        else:
            self.time_limit = self._options["TIME_LIMIT"]

        self._linear_solver = linear_solver

    def solve(self, model, ostream=None):
        """Parameters
        ----------
        model: wntr.aml.Model

        Returns
        -------
        status: SolverStatus
        message: str
        iter_count: int
        """  # noqa: D205
        t0 = time.time()

        x = model.get_x()
        if len(x) == 0:
            return (
                SolverStatus.converged,
                "No variables or constraints",
                0,
            )

        use_r_ = False
        r_ = None
        new_norm = None
        max_absval = 50

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
                print("Success", r)
                return (
                    SolverStatus.converged,
                    "Solved Successfully",
                    outer_iter,
                )

            # get Jacobian
            J = model.evaluate_jacobian(x=None)

            # Call Quantum Linear Solver and get the results
            try:
                # change the range if we can
                if "range" in self._linear_solver.options:
                    self._linear_solver.options["range"] = max_absval
                    print(self._linear_solver.options["range"])

                d = -get_solution_vector(self._linear_solver(J, r))
                print(outer_iter, self._linear_solver, d, r_norm, self.tol)
                max_absval = [5 * x for x in d]

                # d = jacobi_iteration(J.todense(), r, d)
                # print(outer_iter, self._linear_solver, d, r_norm, self.tol)

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
                    msg = f"iter: {outer_iter:<4d} norm: {new_norm:<10.2e} alpha: {alpha:<10.2e} time: {time.time() - t0:<8.4f}"  # noqa: E501
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


def jacobi_iteration(A, b, x0, max_iter=100, eps=1e-3):
    """Jacobi iteration to refine the solution.

    Args:
        A (_type_): _description_
        b (_type_): _description_
        x0 (_type_): _description_
        max_iter (int, optional): _description_. Defaults to 100.
        eps (_type_, optional): _description_. Defaults to 1E-3.
    """
    x = np.copy(x0)
    D = np.diag(A)
    D = np.array([d if d != 0 else 1 for d in D])
    R = A - np.diagflat(D)

    residue = np.linalg.norm(A @ x - b)
    niter = 0

    while residue > eps:
        x = b - np.dot(R, x) / D
        x = np.asarray(x).reshape(-1)
        residue = np.linalg.norm(A @ x - b)
        niter += 1
        if niter > max_iter:
            break

    return x
