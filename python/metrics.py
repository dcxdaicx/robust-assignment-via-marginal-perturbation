from math import log, sqrt

EPSILON = 1e-8


def _candidate_probabilities(instance, probability_matrix):
    for paper, reviewers in enumerate(instance.remained_r_for_p):
        for reviewer in reviewers:
            yield probability_matrix[paper][reviewer]


def quality(instance, probability_matrix, matching_pairs):
    probability_quality = sum(
        probability_matrix[paper][reviewer] * instance.s[paper][reviewer]
        for paper, reviewers in enumerate(instance.remained_r_for_p)
        for reviewer in reviewers
    )
    matching_quality = sum(
        instance.s[paper][reviewer] for paper, reviewer in matching_pairs
    )
    return probability_quality, matching_quality


def maxprob(instance, probability_matrix):
    return max(_candidate_probabilities(instance, probability_matrix), default=0.0)


def avgmaxprob(instance, probability_matrix):
    paper_maxima = [
        max(
            (probability_matrix[paper][reviewer] for reviewer in reviewers),
            default=0.0,
        )
        for paper, reviewers in enumerate(instance.remained_r_for_p)
    ]
    return sum(paper_maxima) / instance.np


def supportsize(instance, probability_matrix):
    return sum(
        probability > EPSILON
        for probability in _candidate_probabilities(instance, probability_matrix)
    )


def entropy(instance, probability_matrix):
    return -sum(
        probability * log(probability)
        for probability in _candidate_probabilities(instance, probability_matrix)
        if probability > EPSILON
    )


def l2normloss(instance, probability_matrix):
    return sqrt(
        sum(
            probability**2
            for probability in _candidate_probabilities(instance, probability_matrix)
        )
    )
