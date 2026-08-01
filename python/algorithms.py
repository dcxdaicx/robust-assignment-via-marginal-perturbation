"""Gurobi implementations of PLRA, PM-Quadratic, and RAMP."""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)
PROGRESS_PAPER_INTERVAL = 5_000
PWL_STEP = 0.1


def _log_paper_progress(stage, paper, total_papers):
    completed = paper + 1
    if completed % PROGRESS_PAPER_INTERVAL == 0 or completed == total_papers:
        logger.info(
            "%s: %s / %s papers",
            stage,
            f"{completed:,}",
            f"{total_papers:,}",
        )


def _pwl_grid(upper_bound):
    """Return a stable PWL grid that includes the exact upper bound."""
    step_count = int(upper_bound / PWL_STEP + 1e-9)
    points = [round(step * PWL_STEP, 10) for step in range(step_count + 1)]
    if points[-1] < upper_bound - 1e-9:
        points.append(float(upper_bound))
    return points


def _add_assignment_and_load_constraints(
    instance, solver, assignment, ellp, senior_only
):
    # Keep the assignment and load constraints aligned across all algorithms.
    for p in range(instance.np):
        assigned = 0
        for r in instance.remained_r_for_p[p]:
            assigned += assignment[p][r]
        if senior_only:
            solver.addConstr(assigned <= ellp[p])
        else:
            solver.addConstr(assigned == ellp[p])

    for r in range(instance.nr):
        load = 0
        for p in instance.remained_p_for_r[r]:
            load += assignment[p][r]
        solver.addConstr(load <= instance.ellr[r])
        if not senior_only or instance.seniority[r] >= 1:
            solver.addConstr(load >= instance.min_ellr[r])


def _assignment_upper_bound(instance, paper, reviewer, maxprob, dynamic_maxprob):
    """Return the probability upper bound for a paper-reviewer pair.

    dynamic_maxprob is now an optional float cap for reviewers who bid on a paper.
    When it is None, we use the global maxprob for every pair.
    """
    if dynamic_maxprob is None:
        return maxprob
    if not (0 < float(dynamic_maxprob) <= 1):
        raise ValueError(f"dynamic_maxprob must be in (0, 1]; got {dynamic_maxprob}")
    if reviewer in instance.bid[paper]:
        return float(dynamic_maxprob)
    return 1.0


def _optimize_and_extract(
    gp, solver, assignment, instance, algorithm_name, senior_only
):
    logger.info("%s model constructed; starting Gurobi optimization", algorithm_name)
    solver.optimize()
    if solver.status == gp.GRB.Status.INFEASIBLE:
        raise RuntimeError(f"{algorithm_name} infeasible; senior_only={senior_only}")
    if solver.status != gp.GRB.Status.OPTIMAL:
        raise RuntimeError(
            f"{algorithm_name} stopped with Gurobi status {solver.status}; "
            f"senior_only={senior_only}"
        )
    logger.info("%s optimization objective: %.6f", algorithm_name, solver.ObjVal)

    for p in range(instance.np):
        for r in instance.remained_r_for_p[p]:
            value = assignment[p][r]
            assignment[p][r] = value.X if isinstance(value, gp.Var) else value
        _log_paper_progress(
            f"Extracting {algorithm_name} solution", p, instance.np
        )
    return assignment


def PLRA(instance, maxprob=1.0, dynamic_maxprob=None, senior_only=False):
    # PLRA Gurobi implementation
    # Inputs arguments:
    #   instance: the input instance
    #   maxprob:  maximum allowed assignment probability
    # Output: an assignment matrix
    # Initialize Gurobi solver
    import gurobipy as gp

    solver = gp.Model()
    solver.setParam("OutputFlag", 0)
    # Initialize assignment matrix and objective function
    objective = 0.0
    assignment = [defaultdict(float) for _ in range(instance.np)]
    if senior_only:
        ellp = instance.ellp_sen.copy()
    else:
        ellp = instance.ellp.copy()
    logger.info("Building PLRA assignment variables and objective")
    for p in range(instance.np):
        for r in instance.remained_r_for_p[p]:
            if instance.constraint_for(p, r) == 1:
                ellp[p] += 1
                assignment[p][r] = 1
                continue
            if instance.constraint_for(p, r) == -1:
                continue
            if senior_only and instance.seniority[r] < 1:
                continue

            assignment[p][r] = solver.addVar(
                lb=0,
                ub=_assignment_upper_bound(instance, p, r, maxprob, dynamic_maxprob),
            )
            objective -= assignment[p][r] * instance.s[p][r]
        _log_paper_progress("Building PLRA variables", p, instance.np)
    solver.setObjective(objective)
    logger.info("Adding PLRA paper and reviewer load constraints")
    _add_assignment_and_load_constraints(
        instance, solver, assignment, ellp, senior_only
    )
    logger.info("Finished adding PLRA assignment and load constraints")
    # Use dual simplex
    solver.params.Method = 1
    return _optimize_and_extract(gp, solver, assignment, instance, "PLRA", senior_only)


def PMQ(instance, beta=0.5, maxprob=1.0, dynamic_maxprob=None, senior_only=False):
    # PM-Quadratic Gurobi implementation
    #   The perturbation function used is f(x) = x - beta * x ^ 2
    # Inputs arguments:
    #   instance: the input instance
    #   beta:     the parameter used in the perturbation function
    #   maxprob:  maximum allowed assignment probability
    # Output: an assignment matrix
    # Initialize Gurobi solver
    import gurobipy as gp

    solver = gp.Model()
    solver.setParam("OutputFlag", 0)
    # Initialize assignment matrix and objective function
    objective = 0.0
    assignment = [defaultdict(float) for _ in range(instance.np)]
    if senior_only:
        ellp = instance.ellp_sen.copy()
    else:
        ellp = instance.ellp.copy()
    logger.info("Building PMQ assignment variables and objective")
    for p in range(instance.np):
        for r in instance.remained_r_for_p[p]:
            if instance.constraint_for(p, r) == 1:
                ellp[p] += 1
                assignment[p][r] = 1
                continue
            if instance.constraint_for(p, r) == -1:
                continue
            if senior_only and instance.seniority[r] < 1:
                continue
            x = solver.addVar(
                lb=0,
                ub=_assignment_upper_bound(instance, p, r, maxprob, dynamic_maxprob),
            )
            assignment[p][r] = x
            objective += (-x + beta * x * x) * instance.s[p][r]
        _log_paper_progress("Building PMQ variables", p, instance.np)
    solver.setObjective(objective)
    logger.info("Adding PMQ paper and reviewer load constraints")
    _add_assignment_and_load_constraints(
        instance, solver, assignment, ellp, senior_only
    )
    logger.info("Finished adding PMQ assignment and load constraints")
    # Use barrier method
    solver.params.Method = 2
    return _optimize_and_extract(gp, solver, assignment, instance, "PMQ", senior_only)


def ramp(
    instance,
    beta=0.5,
    maxprob=1.0,
    dynamic_maxprob=None,
    reward_region=0.1,
    pen_coauthor=0.2,
    pen_2cycle=0.1,
    senior_only=False,
):
    # Our-Algorithm Gurobi implementation
    # Inputs arguments:
    #   instance: the input instance
    #   beta:     the parameter used in the perturbation function as in PMQ
    #   maxprob:  maximum allowed assignment probability
    #   reward_region: region diversity reward
    #   pen_coauthor: coauthor violation penalty
    #   pen_2cycle: 2-cycle violation penalty
    # Output: an assignment matrix
    # Initialize Gurobi solver

    import gurobipy as gp

    solver = gp.Model()
    solver.setParam("OutputFlag", 0)

    # Scale objectives to reduce numerical precision issues.
    obj_scalar = 1000

    # Keep every configured term numerically significant after scaling.
    reward_region = max(reward_region, 1e-3)
    pen_coauthor = max(pen_coauthor, 1e-3)
    pen_2cycle = max(pen_2cycle, 1e-3)

    if senior_only:
        ellp = instance.ellp_sen.copy()
    else:
        ellp = instance.ellp.copy()

    # Create variables
    assignment = [defaultdict(float) for _ in range(instance.np)]
    logger.info("Building RAMP assignment variables")
    for p in range(instance.np):
        for r in instance.remained_r_for_p[p]:
            if instance.constraint_for(p, r) == 1:
                ellp[p] += 1
                assignment[p][r] = 1
                continue
            if instance.constraint_for(p, r) == -1:
                continue
            if senior_only and instance.seniority[r] < 1:
                continue
            assignment[p][r] = solver.addVar(
                lb=0,
                ub=_assignment_upper_bound(instance, p, r, maxprob, dynamic_maxprob),
            )
        _log_paper_progress("Building RAMP variables", p, instance.np)
    objective = 0

    # Set region reward
    logger.info("Adding RAMP region-diversity terms")
    for p in range(instance.np):
        reg_sum = [0 for _ in range(instance.region_count)]
        for r in instance.remained_r_for_p[p]:
            reg_sum[instance.region[r]] += assignment[p][r]
        for reg in range(instance.region_count):
            reg_var = solver.addVar(ub=1)
            solver.addConstr(reg_var <= reg_sum[reg])
            objective -= obj_scalar * reward_region * reg_var
        _log_paper_progress("Adding RAMP region terms", p, instance.np)

    # Set coauthor penalty
    logger.info("Adding RAMP coauthor terms")
    for r in range(instance.nr):
        neighboring_reviewers = instance.coauthorlist[r].copy()
        neighboring_reviewers.append(r)
        for p in instance.bidlist[r]:
            coauthor_var = solver.addVar(lb=1)
            coauthor_sum = 0
            for r2 in neighboring_reviewers:
                coauthor_sum += assignment[p].get(r2, 0.0)
            solver.addConstr(coauthor_var >= coauthor_sum)
            objective += obj_scalar * pen_coauthor * coauthor_var

    # Set linear objective
    # Gurobi needs to first set linear objectives, then set piecewise-linear objectives
    solver.setObjective(objective)

    # Set 2cycle penalty
    logger.info("Adding RAMP strong 2-cycle terms")
    cycle_grids = {}

    def cycle_assignment(paper, reviewer):
        value = assignment[paper].get(reviewer, 0.0)
        if isinstance(value, gp.Var):
            upper_bound = _assignment_upper_bound(
                instance,
                paper,
                reviewer,
                maxprob,
                dynamic_maxprob,
            )
        else:
            upper_bound = float(value)
        return value, upper_bound

    for r1 in range(instance.nr):
        for r2 in instance.bidpaper_authorlist[r1]:
            if r2 > r1 and r1 in instance.bidpaper_author[r2]:
                for p1 in instance.paperlist[r1]:
                    for p2 in instance.paperlist[r2]:
                        if r1 in instance.bid[p2] and r2 in instance.bid[p1]:
                            if (
                                instance.constraint_for(p2, r1) == 1
                                and instance.constraint_for(p1, r2) == 1
                            ):
                                continue
                            first_assignment, first_upper_bound = cycle_assignment(
                                p2, r1
                            )
                            second_assignment, second_upper_bound = cycle_assignment(
                                p1, r2
                            )
                            cycle_upper_bound = (
                                first_upper_bound + second_upper_bound
                            )
                            if cycle_upper_bound == 0:
                                continue
                            sum_2cycle = solver.addVar(
                                lb=0, ub=cycle_upper_bound
                            )
                            solver.addConstr(
                                sum_2cycle
                                >= first_assignment + second_assignment
                            )
                            if cycle_upper_bound not in cycle_grids:
                                xpts = _pwl_grid(cycle_upper_bound)
                                ypts = [
                                    obj_scalar * pen_2cycle * point * point
                                    for point in xpts
                                ]
                                cycle_grids[cycle_upper_bound] = (xpts, ypts)
                            xpts, ypts = cycle_grids[cycle_upper_bound]
                            solver.setPWLObj(sum_2cycle, xpts, ypts)

    # Add piecewise-linear objectives
    logger.info("Adding RAMP piecewise-linear assignment objectives")
    assignment_grids = {}
    for p in range(instance.np):
        for r in instance.remained_r_for_p[p]:
            if isinstance(assignment[p][r], gp.Var):
                upper_bound = _assignment_upper_bound(
                    instance, p, r, maxprob, dynamic_maxprob
                )
                if upper_bound not in assignment_grids:
                    xpts = _pwl_grid(upper_bound)
                    shape = [
                        obj_scalar * (-point + beta * point * point)
                        for point in xpts
                    ]
                    assignment_grids[upper_bound] = (xpts, shape)
                xpts, shape = assignment_grids[upper_bound]
                val = max(1e-3, instance.s[p][r])
                ypts = [value * val for value in shape]
                solver.setPWLObj(assignment[p][r], xpts, ypts)
        _log_paper_progress("Adding RAMP PWL objectives", p, instance.np)

    logger.info("Adding RAMP paper and reviewer load constraints")
    _add_assignment_and_load_constraints(
        instance, solver, assignment, ellp, senior_only
    )
    logger.info("Finished adding RAMP assignment and load constraints")

    # Use dual simplex
    solver.params.Method = 1
    return _optimize_and_extract(gp, solver, assignment, instance, "RAMP", senior_only)
