import logging
import subprocess
from collections import defaultdict
from pathlib import Path

from . import algorithms

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CPP_BINARY_DIR = PROJECT_ROOT / "build" / "bin"
EPSILON = 1e-8


def _solve_fractional_assignment(
    instance,
    algo_name,
    maxprob,
    dynamic_maxprob,
    beta,
    reward_region,
    pen_coauthor,
    pen_2cycle,
    reviewer_pool,
):
    maxprob = 1.0 if maxprob is None else float(maxprob)
    beta = 0.1 if beta is None else float(beta)
    reward_region = 0.2 if reward_region is None else float(reward_region)
    pen_coauthor = 0.1 if pen_coauthor is None else float(pen_coauthor)
    pen_2cycle = 0.2 if pen_2cycle is None else float(pen_2cycle)

    if algo_name == "RAMP":
        return algorithms.ramp(
            instance,
            beta=beta,
            maxprob=maxprob,
            dynamic_maxprob=dynamic_maxprob,
            reward_region=reward_region,
            pen_coauthor=pen_coauthor,
            pen_2cycle=pen_2cycle,
            reviewer_pool=reviewer_pool,
        )
    if algo_name == "Perturbed_Maximization":
        return algorithms.PMQ(
            instance,
            beta=beta,
            maxprob=maxprob,
            dynamic_maxprob=dynamic_maxprob,
            reviewer_pool=reviewer_pool,
        )
    if algo_name == "Randomized":
        return algorithms.PLRA(
            instance,
            maxprob=maxprob,
            dynamic_maxprob=dynamic_maxprob,
            reviewer_pool=reviewer_pool,
        )
    if algo_name == "Default":
        return algorithms.PLRA(
            instance,
            maxprob=1.0,
            dynamic_maxprob=dynamic_maxprob,
            reviewer_pool=reviewer_pool,
        )
    raise ValueError(f"Unknown algorithm: {algo_name}")


def _probability_matrix(instance, solver_assignment):
    """Discard zero and fixed entries without duplicating the dense solution."""
    for paper in range(instance.np):
        solver_assignment[paper] = defaultdict(
            float,
            (
                (reviewer, probability)
                for reviewer, probability in solver_assignment[paper].items()
                if instance.constraint_for(paper, reviewer) == 0
                and probability > EPSILON
            ),
        )
    return solver_assignment


def _round_paper_probabilities(instance, probability_matrix):
    """Round edge weights for the C++ sampler while preserving each paper sum."""
    nonzero_papers_for_reviewer = [[] for _ in range(instance.nr)]
    for paper in range(instance.np):
        nonzero_reviewers = []
        for reviewer, probability in probability_matrix[paper].items():
            probability_matrix[paper][reviewer] = round(probability, 7)
            nonzero_reviewers.append(reviewer)

        total = sum(
            probability_matrix[paper][reviewer] for reviewer in nonzero_reviewers
        )
        difference = round(total) - total
        if abs(difference) > EPSILON:
            for reviewer in nonzero_reviewers:
                current = probability_matrix[paper][reviewer]
                if difference > 0:
                    adjustment = min(difference, 1.0 - current)
                else:
                    adjustment = -min(-difference, current)
                probability_matrix[paper][reviewer] += adjustment
                difference -= adjustment
                if abs(difference) <= EPSILON:
                    break

        for reviewer in nonzero_reviewers:
            if probability_matrix[paper][reviewer] > EPSILON:
                nonzero_papers_for_reviewer[reviewer].append(paper)
    return nonzero_papers_for_reviewer


def _sampler_input(instance, probability_matrix):
    lines = [f"{instance.nr} {instance.np}"]
    lines.extend(
        f"{instance.region[reviewer]} 1" for reviewer in range(instance.nr)
    )
    coauthor_pairs = sorted(
        {
            tuple(sorted((reviewer, coauthor)))
            for reviewer, coauthors in enumerate(instance.coauthorlist)
            for coauthor in coauthors
            if reviewer != coauthor
        }
    )
    lines.append(str(len(coauthor_pairs)))
    lines.extend(
        f"{reviewer} {coauthor}" for reviewer, coauthor in coauthor_pairs
    )

    nonzero_papers_for_reviewer = _round_paper_probabilities(
        instance, probability_matrix
    )
    for reviewer, papers in enumerate(nonzero_papers_for_reviewer):
        lines.extend(
            f"{reviewer} {paper + instance.nr} "
            f"{probability_matrix[paper][reviewer]:.7f}"
            for paper in papers
        )
    return "\n".join(lines) + "\n"


def _sample_matching(instance, optimize_sampling, probability_matrix):
    binary_path = CPP_BINARY_DIR / "bvn"
    if not binary_path.is_file():
        raise RuntimeError(
            f"C++ sampler not found at {binary_path}. Build it with `make cpp`."
        )

    command = [str(binary_path)]
    if optimize_sampling:
        command.append("--optimize-sampling")
    result = subprocess.run(
        command,
        input=_sampler_input(instance, probability_matrix),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "no error output"
        raise RuntimeError(
            f"C++ sampler exited with code {result.returncode}: {detail}"
        )

    matching_pairs = []
    for line_number, line in enumerate(result.stdout.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2:
            raise RuntimeError(
                f"Invalid output from C++ sampler at line {line_number}: {line!r}"
            )
        reviewer_vertex, paper_vertex = map(int, fields)
        reviewer = reviewer_vertex
        paper = paper_vertex - instance.nr
        if not (0 <= paper < instance.np and 0 <= reviewer < instance.nr):
            raise RuntimeError(
                f"Out-of-range pair from C++ sampler at line {line_number}: {line!r}"
            )
        matching_pairs.append([paper, reviewer])
    _validate_matching(instance, probability_matrix, matching_pairs)
    return matching_pairs


def _validate_matching(instance, probability_matrix, matching_pairs):
    """Reject sampler output that violates any hard assignment constraint."""
    expected_paper_loads = [
        round(sum(probabilities.values())) for probabilities in probability_matrix
    ]
    paper_loads = [0] * instance.np
    reviewer_loads = [0] * instance.nr
    fixed_reviewer_loads = [0] * instance.nr
    seen = set()

    for paper, constraints in enumerate(instance.constraint):
        for reviewer, constraint in constraints.items():
            if constraint == 1:
                fixed_reviewer_loads[reviewer] += 1

    for paper, reviewer in matching_pairs:
        pair = (paper, reviewer)
        if pair in seen:
            raise RuntimeError(
                "C++ sampler returned duplicate assignment: "
                f"paper={instance.paper_id(paper)}, "
                f"reviewer={instance.reviewer_id(reviewer)}"
            )
        seen.add(pair)
        if reviewer not in probability_matrix[paper]:
            raise RuntimeError(
                "C++ sampler returned an edge outside the fractional support: "
                f"paper={instance.paper_id(paper)}, "
                f"reviewer={instance.reviewer_id(reviewer)}"
            )
        if instance.constraint_for(paper, reviewer) != 0:
            raise RuntimeError(
                "C++ sampler returned a fixed or forbidden edge: "
                f"paper={instance.paper_id(paper)}, "
                f"reviewer={instance.reviewer_id(reviewer)}"
            )
        paper_loads[paper] += 1
        reviewer_loads[reviewer] += 1

    bad_papers = [
        paper
        for paper, (actual, expected) in enumerate(
            zip(paper_loads, expected_paper_loads)
        )
        if actual != expected
    ]
    if bad_papers:
        paper = bad_papers[0]
        raise RuntimeError(
            "C++ sampler violated paper demand: "
            f"paper={instance.paper_id(paper)}, actual={paper_loads[paper]}, "
            f"expected={expected_paper_loads[paper]}"
        )

    overloaded = [
        reviewer
        for reviewer in range(instance.nr)
        if reviewer_loads[reviewer] + fixed_reviewer_loads[reviewer]
        > instance.ellr[reviewer]
    ]
    if overloaded:
        reviewer = overloaded[0]
        raise RuntimeError(
            "C++ sampler violated reviewer capacity: "
            f"reviewer={instance.reviewer_id(reviewer)}, "
            f"sampled={reviewer_loads[reviewer]}, "
            f"fixed={fixed_reviewer_loads[reviewer]}, "
            f"capacity={instance.ellr[reviewer]}"
        )


def solve(
    instance,
    algo_name,
    maxprob,
    dynamic_maxprob=None,
    beta=0.1,
    reward_region=0.2,
    pen_coauthor=0.1,
    pen_2cycle=0.2,
    optimize_sampling=False,
    reviewer_pool="all",
):
    solver_assignment = _solve_fractional_assignment(
        instance,
        algo_name,
        maxprob,
        dynamic_maxprob,
        beta,
        reward_region,
        pen_coauthor,
        pen_2cycle,
        reviewer_pool,
    )
    probability_matrix = _probability_matrix(instance, solver_assignment)
    matching_pairs = _sample_matching(instance, optimize_sampling, probability_matrix)

    probability_pairs = []
    for paper in range(instance.np):
        for reviewer, probability in probability_matrix[paper].items():
            if probability > EPSILON:
                probability_pairs.append(
                    [paper, reviewer, probability]
                )

    logger.info(
        "Sampled %d assignments from %d non-zero probability edges "
        "(optimization=%s)",
        len(matching_pairs),
        len(probability_pairs),
        optimize_sampling,
    )
    return probability_matrix, probability_pairs, matching_pairs
