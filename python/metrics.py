from math import log, sqrt

EPSILON = 1e-8


def _candidate_probabilities(instance, probability_matrix):
    for probabilities in probability_matrix:
        yield from probabilities.values()


def quality(instance, probability_matrix, matching_pairs):
    probability_quality = sum(
        probability * instance.s[paper][reviewer]
        for paper, probabilities in enumerate(probability_matrix)
        for reviewer, probability in probabilities.items()
    )
    matching_quality = sum(
        instance.s[paper][reviewer] for paper, reviewer in matching_pairs
    )
    return probability_quality, matching_quality


def maxprob(instance, probability_matrix):
    return max(_candidate_probabilities(instance, probability_matrix), default=0.0)


def avgmaxprob(instance, probability_matrix):
    paper_maxima = [
        max(probabilities.values(), default=0.0)
        for probabilities in probability_matrix
    ]
    return sum(paper_maxima) / instance.np


def supportsize(instance, probability_matrix):
    return sum(
        probability > EPSILON
        for probability in _candidate_probabilities(instance, probability_matrix)
    )


def entropy(instance, probability_matrix):
    value = -sum(
        probability * log(probability)
        for probability in _candidate_probabilities(instance, probability_matrix)
        if probability > EPSILON
    )
    return 0.0 if value == 0 else value


def l2normloss(instance, probability_matrix):
    return sqrt(
        sum(
            probability**2
            for probability in _candidate_probabilities(instance, probability_matrix)
        )
    )
