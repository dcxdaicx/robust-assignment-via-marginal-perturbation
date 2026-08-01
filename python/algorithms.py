"""Gurobi implementations of PLRA, PM-Quadratic, and RAMP."""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


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
        if instance.zero_capacity_reviewer_mask[r]:
            continue
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
    solver.setObjective(objective)
    _add_assignment_and_load_constraints(
        instance, solver, assignment, ellp, senior_only
    )
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
    solver.setObjective(objective)
    _add_assignment_and_load_constraints(
        instance, solver, assignment, ellp, senior_only
    )
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
    objective = 0

    # Set region reward
    for p in range(instance.np):
        reg_sum = [0 for _ in range(instance.region_count)]
        for r in instance.remained_r_for_p[p]:
            reg_sum[instance.region[r]] += assignment[p][r]
        for reg in range(instance.region_count):
            reg_var = solver.addVar(ub=1)
            solver.addConstr(reg_var <= reg_sum[reg])
            objective -= obj_scalar * reward_region * reg_var

    # Set coauthor penalty
    for r in range(instance.nr):
        neighboring_reviewers = instance.coauthorlist[r].copy()
        neighboring_reviewers.append(r)
        for p in instance.bidlist[r]:
            coauthor_var = solver.addVar(lb=1)
            coauthor_sum = 0
            for r2 in neighboring_reviewers:
                coauthor_sum += assignment[p][r2]
            solver.addConstr(coauthor_var >= coauthor_sum)
            objective += obj_scalar * pen_coauthor * coauthor_var

    # Set linear objective
    # Gurobi needs to first set linear objectives, then set piecewise-linear objectives
    solver.setObjective(objective)

    # Set 2cycle penalty
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
                            sum_2cycle = solver.addVar(lb=0, ub=1)
                            solver.addConstr(
                                sum_2cycle >= assignment[p2][r1] + assignment[p1][r2]
                            )
                            xpts = []
                            ypts = []
                            now = 0
                            while now <= 1:
                                xpts.append(now)
                                ypts.append(obj_scalar * pen_2cycle * now * now)
                                now += 0.1
                            solver.setPWLObj(sum_2cycle, xpts, ypts)

    # Add piecewise-linear objectives
    for p in range(instance.np):
        for r in instance.remained_r_for_p[p]:
            if isinstance(assignment[p][r], gp.Var):
                xpts = []
                ypts = []
                now = 0
                val = max(1e-3, instance.s[p][r])
                while now <= maxprob + (1e-6):
                    xpts.append(now)
                    ypts.append(obj_scalar * (-now + beta * now * now) * val)
                    now += 0.1
                solver.setPWLObj(assignment[p][r], xpts, ypts)

    _add_assignment_and_load_constraints(
        instance, solver, assignment, ellp, senior_only
    )

    # Use dual simplex
    solver.params.Method = 1
    return _optimize_and_extract(gp, solver, assignment, instance, "RAMP", senior_only)
