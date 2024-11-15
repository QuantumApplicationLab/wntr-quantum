import logging
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from quantum_newton_raphson.splu_solver import SPLU_SOLVER
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
        result (SPLU_SOLVER | VQLSResult | QUBOResult | HHLresult): linear solver dataclass result

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

        if "FIXED_POINT" not in self._options:
            self.fixed_point = False
        else:
            self.fixed_point = self._options["FIXED_POINT"]

        self._linear_solver = linear_solver

    def solve(self, model, ostream=None, debug=False):
        """Parameters
        ----------
        model: wntr.aml.Model

        Returns
        -------
        status: SolverStatus
        message: str
        iter_count: int
        linear_solver_results: list
        """  # noqa: D205
        linear_solver_results = []

        t0 = time.time()

        x = model.get_x()
        if len(x) == 0:
            return (
                SolverStatus.converged,
                "No variables or constraints",
                0,
                linear_solver_results,
            )

        use_r_ = False
        r_ = None
        new_norm = None
        ref_solver = SPLU_SOLVER()

        if hasattr(self._linear_solver, "options"):
            if "range" in self._linear_solver.options:
                initial_range = self._linear_solver.options["range"]
                initial_offset = self._linear_solver.options["offset"]

        # MAIN NEWTON LOOP
        for outer_iter in range(self.maxiter):
            if time.time() - t0 >= self.time_limit:
                return (
                    SolverStatus.error,
                    "Time limit exceeded",
                    outer_iter,
                    linear_solver_results,
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
                    linear_solver_results,
                )

            # get Jacobian
            J = model.evaluate_jacobian(x=None)

            # Call Quantum Linear Solver and get the results
            try:

                print("Matrix")
                print(J.todense())
                print("RHS")
                print(r)

                # get the reference solution
                dref = -get_solution_vector(ref_solver(J, r))
                print(dref)

                # get the approxmate of the solution
                approx_result = self._linear_solver(J, r)
                d = -get_solution_vector(approx_result)

                # save linear solver result
                linear_solver_results.append(approx_result)
                # print(outer_iter, self._linear_solver, d, r_norm, self.tol)
                print(d)

                if debug:
                    plt.scatter(dref, d)
                    plt.axline((0, 0), slope=1, color="k")
                    plt.show()

                # use a fixed point [doesn't work]
                if self.fixed_point:
                    d = sor_solver(
                        (J.T @ J).todense(), (J @ r), d, max_iter=10, eps=1e-3
                    )

                # init the range encoding
                if hasattr(self._linear_solver, "options"):
                    if "range" in self._linear_solver.options:
                        self._linear_solver.options["range"] = initial_range
                        self._linear_solver.options["offset"] = initial_offset

            except sp.linalg.MatrixRankWarning:
                return (
                    SolverStatus.error,
                    "Jacobian is singular at iteration " + str(outer_iter),
                    outer_iter,
                    linear_solver_results,
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
                        linear_solver_results,
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
            linear_solver_results,
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


def sor_solver(A, b, initial_guess, max_iter=10, omega=0.05, eps=1e-8):
    """This is an implementation of the pseduo-code provided in the Wikipedia article.
    Inputs:
      A: nxn numpy matrix
      b: n dimensional numpy vector
      omega: relaxation factor
      initial_guess: An initial solution guess for the solver to start with
    Returns:
      phi: solution vector of dimension n.
    """  # noqa: D205
    phi = np.copy(initial_guess)
    residual = np.linalg.norm(np.matmul(A, phi) - b)  # Initial residual
    iiter = 0
    while residual > eps:
        if iiter >= max_iter:
            break
        for i in range(A.shape[0]):
            sigma = 0
            for j in range(A.shape[1]):
                if j != i:
                    sigma += A[i, j] * phi[j]
            phi[i] = (1 - omega) * phi[i] + (omega / A[i, i]) * (b[i] - sigma)
        residual = np.linalg.norm(np.matmul(A, phi) - b)
        # print("Residual: {0:10.6g}".format(residual))
        iiter += 1
    return phi
