import logging
import subprocess
from collections import defaultdict
from math import floor, isfinite
from pathlib import Path

from . import algorithms

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CPP_BINARY_DIR = PROJECT_ROOT / "build" / "bin"
EPSILON = 1e-8
PROBABILITY_SCALE = 10000000


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


def _round_paper_probabilities(
    instance, probability_matrix, maxprob=1.0, dynamic_maxprob=None
):
    """Quantize probabilities while preserving edge, paper, and reviewer bounds."""
    fixed_reviewer_loads = [0] * instance.nr
    for constraints in instance.constraint:
        for reviewer, constraint in constraints.items():
            if constraint == 1:
                fixed_reviewer_loads[reviewer] += 1

    reviewer_unit_capacities = [
        (instance.ellr[reviewer] - fixed_reviewer_loads[reviewer])
        * PROBABILITY_SCALE
        for reviewer in range(instance.nr)
    ]
    reviewer_base_units = [0] * instance.nr
    base_units_by_paper = []
    increment_candidates = []
    paper_deficits = []

    for paper in range(instance.np):
        base_units = {}
        candidates = []
        probability_sum = 0.0
        for reviewer, probability in probability_matrix[paper].items():
            if not isfinite(probability) or probability < 0:
                raise RuntimeError(
                    f"Invalid fractional probability at ({paper}, {reviewer}): "
                    f"{probability}"
                )
            probability_sum += probability
            scaled = probability * PROBABILITY_SCALE
            edge_cap = algorithms._assignment_upper_bound(
                instance,
                paper,
                reviewer,
                maxprob,
                dynamic_maxprob,
            )
            maximum_units = round(edge_cap * PROBABILITY_SCALE)
            if scaled > maximum_units + EPSILON * PROBABILITY_SCALE:
                raise RuntimeError(
                    "Fractional solution exceeds an assignment-probability cap: "
                    f"paper={instance.paper_id(paper)}, "
                    f"reviewer={instance.reviewer_id(reviewer)}, "
                    f"probability={probability}, cap={edge_cap}"
                )
            floored_units = floor(scaled)
            units = min(floored_units, maximum_units)
            base_units[reviewer] = units
            reviewer_base_units[reviewer] += units
            fractional_part = scaled - floored_units
            if fractional_part > 1e-9 and units < maximum_units:
                candidates.append((reviewer, fractional_part))

        expected_paper_load = round(probability_sum)
        if abs(probability_sum - expected_paper_load) > 1e-5:
            raise RuntimeError(
                "Fractional solution does not have an integral paper sum: "
                f"paper={instance.paper_id(paper)}, total={probability_sum}"
            )
        deficit = expected_paper_load * PROBABILITY_SCALE - sum(base_units.values())
        if deficit < 0 or deficit > len(candidates):
            raise RuntimeError(
                "Unable to quantize a paper row within its edge bounds: "
                f"paper={instance.paper_id(paper)}, deficit={deficit}, "
                f"roundable_edges={len(candidates)}"
            )
        candidates.sort(key=lambda item: (-item[1], item[0]))
        base_units_by_paper.append(base_units)
        increment_candidates.append([reviewer for reviewer, _ in candidates])
        paper_deficits.append(deficit)

    reviewer_increment_capacities = [
        capacity - base
        for capacity, base in zip(reviewer_unit_capacities, reviewer_base_units)
    ]
    overloaded = [
        reviewer
        for reviewer, capacity in enumerate(reviewer_increment_capacities)
        if capacity < 0
    ]
    if overloaded:
        reviewer = overloaded[0]
        raise RuntimeError(
            "Fractional solution exceeds reviewer capacity before quantization: "
            f"reviewer={instance.reviewer_id(reviewer)}"
        )

    selected_by_paper = [set() for _ in range(instance.np)]
    selected_papers_by_reviewer = [set() for _ in range(instance.nr)]
    reviewer_increment_counts = [0] * instance.nr

    def select(paper, reviewer):
        selected_by_paper[paper].add(reviewer)
        selected_papers_by_reviewer[reviewer].add(paper)
        reviewer_increment_counts[reviewer] += 1

    def deselect(paper, reviewer):
        selected_by_paper[paper].remove(reviewer)
        selected_papers_by_reviewer[reviewer].remove(paper)
        reviewer_increment_counts[reviewer] -= 1

    def augment(paper, seen_papers, seen_reviewers):
        for reviewer in increment_candidates[paper]:
            if reviewer in selected_by_paper[paper] or reviewer in seen_reviewers:
                continue
            seen_reviewers.add(reviewer)
            if (
                reviewer_increment_counts[reviewer]
                < reviewer_increment_capacities[reviewer]
            ):
                select(paper, reviewer)
                return True
            for other_paper in tuple(selected_papers_by_reviewer[reviewer]):
                if other_paper in seen_papers:
                    continue
                seen_papers.add(other_paper)
                if augment(other_paper, seen_papers, seen_reviewers):
                    deselect(other_paper, reviewer)
                    select(paper, reviewer)
                    return True
        return False

    paper_order = sorted(
        range(instance.np),
        key=lambda paper: (
            len(increment_candidates[paper]) - paper_deficits[paper],
            len(increment_candidates[paper]),
        ),
    )
    for paper in paper_order:
        for _ in range(paper_deficits[paper]):
            if not augment(paper, {paper}, set()):
                raise RuntimeError(
                    "Unable to preserve reviewer capacities while quantizing "
                    f"paper={instance.paper_id(paper)}"
                )

    nonzero_papers_for_reviewer = [[] for _ in range(instance.nr)]
    for paper, base_units in enumerate(base_units_by_paper):
        quantized = defaultdict(float)
        for reviewer, units in base_units.items():
            units += reviewer in selected_by_paper[paper]
            if units:
                quantized[reviewer] = units / PROBABILITY_SCALE
                nonzero_papers_for_reviewer[reviewer].append(paper)
        probability_matrix[paper] = quantized
    return nonzero_papers_for_reviewer


def _sampler_input(
    instance, probability_matrix, maxprob=1.0, dynamic_maxprob=None
):
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
        instance, probability_matrix, maxprob, dynamic_maxprob
    )
    for reviewer, papers in enumerate(nonzero_papers_for_reviewer):
        lines.extend(
            f"{reviewer} {paper + instance.nr} "
            f"{probability_matrix[paper][reviewer]:.7f}"
            for paper in papers
        )
    return "\n".join(lines) + "\n"


def _sample_matching(
    instance,
    optimize_sampling,
    probability_matrix,
    maxprob=1.0,
    dynamic_maxprob=None,
):
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
        input=_sampler_input(
            instance, probability_matrix, maxprob, dynamic_maxprob
        ),
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
    matching_pairs = _sample_matching(
        instance,
        optimize_sampling,
        probability_matrix,
        maxprob,
        dynamic_maxprob,
    )

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
