"""
===========================
Mathematical things.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2017
---------------------------
"""

from enum import Enum, auto

import numpy
import nltk

from scipy import spatial


class CorrelationType(Enum):
    """
    Representative of a correlation type.
    """
    Pearson  = auto()
    Spearman = auto()


class DistanceType(Enum):
    """
    Representative of a distance type.
    """
    correlation = auto()
    cosine      = auto()
    Euclidean   = auto()

    @property
    def name(self) -> str:
        """
        A string representation of the distance type.
        """
        if self is DistanceType.Euclidean:
            return "Euclidean"
        elif self is DistanceType.cosine:
            return "cosine"
        elif self is DistanceType.correlation:
            return "correlation"
        else:
            raise ValueError()


def distance(u: numpy.ndarray, v: numpy.ndarray, distance_type: DistanceType) -> float:
    """
    Distance from vector u to vector v using the specified distance type.
    """

    if distance_type is DistanceType.Euclidean:
        return _euclidean_distance(u, v)
    elif distance_type is DistanceType.cosine:
        return _cosine_distance(u, v)
    elif distance_type is DistanceType.correlation:
        return _correlation_distance(u, v)
    else:
        raise ValueError()


def _euclidean_distance(u: numpy.ndarray, v: numpy.ndarray):
    """
    Euclidean distance.
    :param u:
    :param v:
    :return:
    """
    return numpy.linalg.norm(u - v)


def _cosine_distance(u: numpy.ndarray, v: numpy.ndarray):
    """
    Cosine distance.
    :param u:
    :param v:
    :return:
    """
    return spatial.distance.cosine(u, v)


def _correlation_distance(u: numpy.ndarray, v: numpy.ndarray):
    """
    Correlation distance.
    :param u:
    :param v:
    :return:
    """
    r = numpy.corrcoef(u, v)[0, 1]
    return 1 - r


def sparse_max(a, b):
    """
    Element-wise maximum for same-sized sparse matrices.
    Thanks to https://stackoverflow.com/a/19318259/2883198
    """

    # Where are elements of b bigger than corresponding element of a?
    b_is_bigger = a - b
    # Pycharm gets type inference wrong here, I'm pretty sure
    # noinspection PyTypeChecker
    b_is_bigger.data = numpy.where(b_is_bigger.data < 0, 1, 0)

    # Return elements of a where a was bigger, and elements of b where b was bigger
    return a - a.multiply(b_is_bigger) + b.multiply(b_is_bigger)


def levenshtein_distance(string_1: str, string_2: str) -> float:
    """
    Levenshtein edit distance between two strings.
    """
    return nltk.edit_distance(string_1, string_2,
                              substitution_cost=1,
                              transpositions=False)
