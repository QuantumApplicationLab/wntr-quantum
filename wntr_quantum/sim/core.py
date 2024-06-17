import logging
import warnings
import wntr.sim.hydraulics
import wntr.sim.results
from quantum_newton_raphson.splu_solver import SPLU_SOLVER
from wntr.sim.core import WNTRSimulator
from wntr.sim.core import _Diagnostics
from wntr.sim.core import _ValveSourceChecker
from .solvers import QuantumNewtonSolver

logger = logging.getLogger(__name__)


class QuantumWNTRSimulator(WNTRSimulator):
    """The quantum enabled NR slver."""

    def __init__(self, wn, linear_solver=SPLU_SOLVER()):  # noqa: D417
        """WNTR simulator class.
        The WNTR simulator uses a custom newton solver and linear solvers from scipy.sparse.

        Parameters
        ----------
        wn : WaterNetworkModel object
            Water network model
        linear_solver: The linear solver used for the NR step


        .. note::
            The mode parameter has been deprecated. Please set the mode using Options.hydraulic.demand_model

        """  # noqa: D205
        super().__init__(wn)
        self._linear_solver = linear_solver
        self._solver = QuantumNewtonSolver(linear_solver=linear_solver)

    def run_sim(
        self,
        solver=QuantumNewtonSolver,
        linear_solver=SPLU_SOLVER(),
        backup_solver=None,
        solver_options=None,
        backup_solver_options=None,
        convergence_error=False,
        HW_approx="default",
        diagnostics=False,
    ):
        """Run an extended period simulation (hydraulics only).

        Parameters
        ----------
        solver: object
            wntr.sim.solvers.NewtonSolver or Scipy solver
        linear_solver: linear solver
            Linear solver
        backup_solver: object
            wntr.sim.solvers.NewtonSolver or Scipy solver
        solver_options: dict
            Solver options are specified using the following dictionary keys:
        backup_solver_options: dict
            Solver options are specified using the following dictionary keys:

            * MAXITER: the maximum number of iterations for each hydraulic solve
                (each timestep and trial) (default = 3000)
            * TOL: tolerance for the hydraulic equations (default = 1e-6)
            * BT_RHO: the fraction by which the step length is reduced at each iteration of the
                line search (default = 0.5)
            * BT_MAXITER: the maximum number of iterations for each line search (default = 100)
            * BACKTRACKING: whether or not to use a line search (default = True)
            * BT_START_ITER: the newton iteration at which a line search should start being used (default = 2)
            * THREADS: the number of threads to use in constraint and jacobian computations
        backup_solver_options: dict
        convergence_error: bool (optional)
            If convergence_error is True, an error will be raised if the
            simulation does not converge. If convergence_error is False, partial results are returned,
            a warning will be issued, and results.error_code will be set to 0
            if the simulation does not converge.  Default = False.
        HW_approx: str
            Specifies which Hazen-Williams headloss approximation to use. Options are 'default' and 'piecewise'. Please
            see the WNTR documentation on hydraulics for details.
        diagnostics: bool
            If True, then run with diagnostics on
        """
        self._linear_solver = linear_solver

        logger.debug('creating hydraulic model')
        self.mode = self._wn.options.hydraulic.demand_model
        self._model, self._model_updater = wntr.sim.hydraulics.create_hydraulic_model(wn=self._wn, HW_approx=HW_approx)

        if diagnostics:
            diagnostics = _Diagnostics(self._wn, self._model, self.mode, enable=True)
        else:
            diagnostics = _Diagnostics(self._wn, self._model, self.mode, enable=False)

        self._setup_sim_options(solver=solver, backup_solver=backup_solver, solver_options=solver_options,
                                backup_solver_options=backup_solver_options, convergence_error=convergence_error)

        self._valve_source_checker = _ValveSourceChecker(self._wn)
        self._get_control_managers()
        self._register_controls_with_observers()

        node_res, link_res = wntr.sim.hydraulics.initialize_results_dict(self._wn)
        results = wntr.sim.results.SimulationResults()
        results.error_code = None
        results.time = []
        results.network_name = self._wn.name

        self._initialize_internal_graph()
        self._change_tracker.set_reference_point('graph')
        self._change_tracker.set_reference_point('model')

        if self._wn.sim_time == 0:
            first_step = True
        else:
            first_step = False
        trial = -1
        max_trials = self._wn.options.hydraulic.trials
        resolve = False
        self._rule_iter = 0  # this is used to determine the rule timestep

        if first_step:
            wntr.sim.hydraulics.update_network_previous_values(self._wn)
            self._wn._prev_sim_time = -1

        logger.debug('starting simulation')

        logger.info('{0:<10}{1:<10}{2:<10}{3:<15}{4:<15}'.format(
            'Sim Time', 'Trial', 'Solver', '# isolated', '# isolated')
        )
        logger.info('{0:<10}{1:<10}{2:<10}{3:<15}{4:<15}'.format('', '', '# iter', 'junctions', 'links'))
        while True:
            if logger.getEffectiveLevel() <= logging.DEBUG:
                logger.debug('\n\n')

            if not resolve:
                if not first_step:
                    """
                    The tank levels/heads must be done before checking the controls because the TankLevelControls
                    depend on the tank levels. These will be updated again after we determine the next actual timestep.
                    """
                    wntr.sim.hydraulics.update_tank_heads(self._wn)
                trial = 0
                self._compute_next_timestep_and_run_presolve_controls_and_rules(first_step)

            self._run_feasibility_controls()

            # Prepare for solve
            self._update_internal_graph()
            num_isolated_junctions, num_isolated_links = self._get_isolated_junctions_and_links()
            if not first_step and not resolve:
                wntr.sim.hydraulics.update_tank_heads(self._wn)
            wntr.sim.hydraulics.update_model_for_controls(
                self._model, self._wn, self._model_updater, self._change_tracker
            )
            wntr.sim.models.param.source_head_param(self._model, self._wn)
            wntr.sim.models.param.expected_demand_param(self._model, self._wn)

            diagnostics.run(last_step='presolve controls, rules, and model updates', next_step='solve')

            solver_status, mesg, iter_count = _solver_helper(
                self._model, self._solver, self._linear_solver, self._solver_options
            )
            if solver_status == 0 and self._backup_solver is not None:
                solver_status, mesg, iter_count = _solver_helper(
                    self._model, self._backup_solver, self._linear_solver, self._backup_solver_options
                )
            if solver_status == 0:
                if self._convergence_error:
                    logger.error('Simulation did not converge at time ' + self._get_time() + '. ' + mesg)
                    raise RuntimeError('Simulation did not converge at time ' + self._get_time() + '. ' + mesg)
                warnings.warn('Simulation did not converge at time ' + self._get_time() + '. ' + mesg)
                logger.warning('Simulation did not converge at time ' + self._get_time() + '. ' + mesg)
                results.error_code = wntr.sim.results.ResultsStatus.error
                diagnostics.run(last_step='solve', next_step='break')
                break

            logger.info('{0:<10}{1:<10}{2:<10}{3:<15}{4:<15}'.format(
                self._get_time(), trial, iter_count, num_isolated_junctions, num_isolated_links
            ))

            # Enter results in network and update previous inputs
            logger.debug('storing results in network')
            wntr.sim.hydraulics.store_results_in_network(self._wn, self._model)

            diagnostics.run(last_step='solve and store results in network', next_step='postsolve controls')

            self._run_postsolve_controls()
            self._run_feasibility_controls()
            if self._change_tracker.changes_made(ref_point='graph'):
                resolve = True
                self._update_internal_graph()
                wntr.sim.hydraulics.update_model_for_controls(
                    self._model, self._wn, self._model_updater, self._change_tracker
                )
                diagnostics.run(last_step='postsolve controls and model updates', next_step='solve next trial')
                trial += 1
                if trial > max_trials:
                    if convergence_error:
                        logger.error('Exceeded maximum number of trials at time ' + self._get_time() + '. ')
                        raise RuntimeError('Exceeded maximum number of trials at time ' + self._get_time() + '. ' )
                    results.error_code = wntr.sim.results.ResultsStatus.error
                    warnings.warn('Exceeded maximum number of trials at time ' + self._get_time() + '. ')
                    logger.warning('Exceeded maximum number of trials at time ' + self._get_time() + '. ' )
                    break
                continue

            diagnostics.run(last_step='postsolve controls and model updates', next_step='advance time')

            logger.debug('no changes made by postsolve controls; moving to next timestep')

            resolve = False
            if isinstance(self._report_timestep, (float, int)):
                if self._wn.sim_time % self._report_timestep == 0:
                    wntr.sim.hydraulics.save_results(self._wn, node_res, link_res)
                    if len(results.time) > 0 and int(self._wn.sim_time) == results.time[-1]:
                        if int(self._wn.sim_time) != self._wn.sim_time:
                            raise RuntimeError('Time steps increments smaller than 1 second are forbidden.'+
                                               ' Keep time steps as an integer number of seconds.')
                        else:
                            raise RuntimeError('Simulation already solved this timestep')
                    results.time.append(int(self._wn.sim_time))
            elif self._report_timestep.upper() == 'ALL':
                wntr.sim.hydraulics.save_results(self._wn, node_res, link_res)
                if len(results.time) > 0 and int(self._wn.sim_time) == results.time[-1]:
                    raise RuntimeError('Simulation already solved this timestep')
                results.time.append(int(self._wn.sim_time))
            wntr.sim.hydraulics.update_network_previous_values(self._wn)
            first_step = False
            self._wn.sim_time += self._hydraulic_timestep
            overstep = float(self._wn.sim_time) % self._hydraulic_timestep
            self._wn.sim_time -= overstep

            if self._wn.sim_time > self._wn.options.time.duration:
                break

        wntr.sim.hydraulics.get_results(self._wn, results, node_res, link_res)

        return results


def _solver_helper(model, solver, linear_solver, solver_options):
    """Parameters
    ----------
    model: wntr.aml.Model
    solver: class or function
    solver_options: dict

    Returns
    -------
    solver_status: int
    message: str
    """  # noqa: D205
    logger.debug("solving")
    model.set_structure()
    if solver is QuantumNewtonSolver:
        _solver = QuantumNewtonSolver(linear_solver, options=solver_options)
        sol = _solver.solve(model)
    else:
        raise ValueError("Solver not recognized.")
    return sol
