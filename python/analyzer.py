from collections import defaultdict
from contextlib import redirect_stdout
import logging
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace

import matplotlib.pyplot as plt

from . import metrics

logger = logging.getLogger(__name__)


def analyze(instance, prob_assignment_matrix, matching_pairs, final=True):
    statistics = SimpleNamespace()

    statistics.prob_quality, statistics.matching_quality = metrics.quality(
        instance, prob_assignment_matrix, matching_pairs
    )
    statistics.max_prob = metrics.maxprob(instance, prob_assignment_matrix)
    statistics.avg_max_prob = metrics.avgmaxprob(instance, prob_assignment_matrix)
    statistics.support_size = metrics.supportsize(instance, prob_assignment_matrix)
    statistics.entropy = metrics.entropy(instance, prob_assignment_matrix)
    statistics.l2norm_loss = metrics.l2normloss(instance, prob_assignment_matrix)

    if not final:
        return statistics

    final_matching_r_for_p = [set() for _ in range(instance.np)]
    stage_matching_r_for_p = [set() for _ in range(instance.np)]

    for paper, constraints in enumerate(instance.constraint):
        for reviewer, constraint in constraints.items():
            if constraint == 1:
                final_matching_r_for_p[paper].add(reviewer)
    for paper, reviewers in enumerate(instance.matched_this_stage):
        stage_matching_r_for_p[paper].update(reviewers)

    for paper, reviewer in matching_pairs:
        final_matching_r_for_p[paper].add(reviewer)
        stage_matching_r_for_p[paper].add(reviewer)

    diversity_scores = []
    for assignments in final_matching_r_for_p:
        if not assignments:
            continue
        unique_regions = len(set(instance.region[r] for r in assignments))
        diversity_scores.append(unique_regions / len(assignments))
    statistics.geographic_diversity = (
        sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0
    )

    required_papers = 0
    satisfied_papers = 0
    for p in range(instance.np):
        if not instance.ellp_sen[p]:
            continue
        required_papers += 1
        senior_reviewers = 0
        for r in stage_matching_r_for_p[p]:
            if instance.seniority[r] >= 1:
                senior_reviewers += 1
        if senior_reviewers >= instance.ellp_sen[p]:
            satisfied_papers += 1
    statistics.seniority_requirement_fulfillment_rate = (
        1 if required_papers == 0 else satisfied_papers / required_papers
    )

    statistics.coauthor_dist_1_pairs_num = 0
    statistics.coauthor_dist_2_pairs_num = 0
    for p in range(instance.np):
        for r1 in final_matching_r_for_p[p]:
            seen_distance_2 = defaultdict(bool)
            for r2 in instance.coauthorlist[r1]:
                if r2 > r1:
                    statistics.coauthor_dist_1_pairs_num += (
                        r2 in final_matching_r_for_p[p]
                    )
                for r3 in instance.coauthorlist[r2]:
                    if (
                        r3 <= r1
                        or r3 in instance.coauthorship[r1]
                        or seen_distance_2[r3]
                    ):
                        continue
                    seen_distance_2[r3] = True
                    statistics.coauthor_dist_2_pairs_num += (
                        r3 in final_matching_r_for_p[p]
                    )

    statistics.strong_twocycle_violations_1 = 0
    for r1 in range(instance.nr):
        for r2 in instance.bidpaper_authorlist[r1]:
            if r2 > r1 and r1 in instance.bidpaper_author[r2]:
                for p1 in instance.paperlist[r1]:
                    for p2 in instance.paperlist[r2]:
                        if r1 in instance.bid[p2] and r2 in instance.bid[p1]:
                            if (
                                r2 in final_matching_r_for_p[p1]
                                and r1 in final_matching_r_for_p[p2]
                            ):
                                statistics.strong_twocycle_violations_1 += 1
                                logger.warning(
                                    "Strong 2-cycle between reviewers %s and %s on papers %s and %s",
                                    instance.reviewer_id(r1),
                                    instance.reviewer_id(r2),
                                    instance.paper_id(p1),
                                    instance.paper_id(p2),
                                )

    reviewer_loads = [0 for _ in range(instance.nr)]
    for p in range(instance.np):
        for r in final_matching_r_for_p[p]:
            reviewer_loads[r] += 1

    max_load = max(reviewer_loads) if reviewer_loads else 0
    statistics.load_distribution = [0 for _ in range(max_load + 1)]
    for load in reviewer_loads:
        statistics.load_distribution[load] += 1

    statistics.matched_pair_scores = []
    statistics.assigned_bid_scores = []
    for p in range(instance.np):
        for r in final_matching_r_for_p[p]:
            statistics.matched_pair_scores.append(instance.s[p].get(r, 0.0))
            statistics.assigned_bid_scores.append(instance.bid_score[p].get(r, 0.0))

    statistics.reviewer_bid_coverage = []
    for r in range(instance.nr):
        if reviewer_loads[r] == 0:
            continue
        bid_covered_assignments = 0
        for p, reviewers in enumerate(final_matching_r_for_p):
            if r in reviewers and instance.bid_score[p].get(r, 0.0) > 0:
                bid_covered_assignments += 1
        statistics.reviewer_bid_coverage.append(
            bid_covered_assignments / reviewer_loads[r]
        )

    return statistics


def output_statistics(instance, statistics, output_dir, algo_name):
    output_path = Path(output_dir)
    stats_path = output_path / f"{algo_name}_statistics.txt"
    with open(stats_path, "w", encoding="utf-8") as file:
        with redirect_stdout(file):
            print(f"running_time: {statistics.running_time:.3f} seconds")
            print(f"total papers: {instance.np}")
            print(f"total reviewers: {instance.nr}")
            if hasattr(instance, "max_quality"):
                print("max_quality:", f"{instance.max_quality:.3f}")
            print("prob_assignment_quality:", f"{statistics.prob_quality:.3f}")
            print("matching_pairs_quality:", f"{statistics.matching_quality:.3f}")
            if hasattr(instance, "max_quality"):
                print(
                    "prob_assignment_quality_relative:",
                    f"{statistics.prob_quality / instance.max_quality:.3f}",
                )
                print(
                    "matching_pairs_quality_relative:",
                    f"{statistics.matching_quality / instance.max_quality:.3f}",
                )
            print("support_size:", statistics.support_size)
            print("entropy:", f"{statistics.entropy:.3f}")
            print("geographic_diversity:", f"{statistics.geographic_diversity:.3f}")
            print(
                "seniority_requirement_fulfillment_rate:",
                f"{statistics.seniority_requirement_fulfillment_rate:.3f}",
            )
            print("coauthor_dist_1_pairs:", statistics.coauthor_dist_1_pairs_num)
            print("coauthor_dist_2_pairs:", statistics.coauthor_dist_2_pairs_num)
            print(
                "strong_twocycle_violations:", statistics.strong_twocycle_violations_1
            )
            print("load distribution:", statistics.load_distribution)

    score_plot_path = output_path / "matched_score_distribution.png"
    plt.figure(figsize=(10, 6))
    if statistics.matched_pair_scores:
        plt.hist(
            statistics.matched_pair_scores,
            bins=20,
            color="lightblue",
            edgecolor="gray",
            alpha=0.8,
        )
        score_mean = mean(statistics.matched_pair_scores)
        score_std = (
            pstdev(statistics.matched_pair_scores)
            if len(statistics.matched_pair_scores) > 1
            else 0.0
        )
        plt.axvline(score_mean, color="blue", linestyle="--", linewidth=2)
        plt.text(
            0.03,
            0.92,
            f"mu = {score_mean:.3f}\nsigma = {score_std:.3f}",
            transform=plt.gca().transAxes,
            bbox=dict(facecolor="white", edgecolor="blue", alpha=0.9),
            fontsize=12,
        )
    plt.title("Aggregate Scores of Matched Pairs", fontsize=18, fontweight="bold")
    plt.xlabel("Score", fontsize=14)
    plt.ylabel("Number of Matched Pairs", fontsize=14)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(score_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    load_plot_path = output_path / "reviewer_load_distribution.png"
    plt.figure(figsize=(10, 6))
    load_values = list(range(len(statistics.load_distribution)))
    plt.bar(
        load_values,
        statistics.load_distribution,
        color="steelblue",
        edgecolor="black",
        alpha=0.85,
    )
    plt.title("Reviewer Load Distribution", fontsize=18, fontweight="bold")
    plt.xlabel("Assigned Papers", fontsize=14)
    plt.ylabel("Number of Reviewers", fontsize=14)
    plt.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(load_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    bid_plot_path = output_path / "assigned_bid_distribution.png"
    plt.figure(figsize=(10, 6))
    if statistics.assigned_bid_scores:
        bid_values = sorted(set(statistics.assigned_bid_scores))
        if len(bid_values) <= 20:
            bid_counts = []
            for bid_value in bid_values:
                bid_counts.append(
                    sum(
                        1
                        for score in statistics.assigned_bid_scores
                        if score == bid_value
                    )
                )
            plt.bar(
                [str(value) for value in bid_values],
                bid_counts,
                color="steelblue",
                edgecolor="black",
                alpha=0.85,
            )
            for index, count in enumerate(bid_counts):
                plt.text(
                    index, count, str(count), ha="center", va="bottom", fontsize=10
                )
        else:
            plt.hist(
                statistics.assigned_bid_scores,
                bins=20,
                color="steelblue",
                edgecolor="black",
                alpha=0.85,
            )
    plt.title("Bid Distribution of Assigned Pairs", fontsize=18, fontweight="bold")
    plt.xlabel("Bid Score", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(bid_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    coverage_plot_path = output_path / "reviewer_bid_coverage_distribution.png"
    plt.figure(figsize=(10, 6))
    if statistics.reviewer_bid_coverage:
        plt.hist(
            statistics.reviewer_bid_coverage,
            bins=10,
            color="steelblue",
            edgecolor="black",
            alpha=0.85,
        )
        coverage_mean = mean(statistics.reviewer_bid_coverage)
        coverage_std = (
            pstdev(statistics.reviewer_bid_coverage)
            if len(statistics.reviewer_bid_coverage) > 1
            else 0.0
        )
        plt.text(
            0.03,
            0.90,
            f"N = {len(statistics.reviewer_bid_coverage)}\nmu = {coverage_mean:.3f}\nsigma = {coverage_std:.3f}",
            transform=plt.gca().transAxes,
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.9),
            fontsize=12,
        )
    plt.title(
        "Reviewer Bid Coverage Among Assigned Papers", fontsize=18, fontweight="bold"
    )
    plt.xlabel("Fraction of Assigned Papers Bid On by Reviewer", fontsize=14)
    plt.ylabel("Number of Reviewers", fontsize=14)
    plt.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(coverage_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
