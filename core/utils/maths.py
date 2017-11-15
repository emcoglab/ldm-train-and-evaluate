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

import math
from enum import Enum, auto

import nltk
import numpy
from scipy import spatial, integrate


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


def magnitude_of_negative(c: float) -> float:
    """
    Returns the absolute value of input `c` when it is negative, and 0 otherwise.
    """
    # If c negative
    if c < 0:
        # Make it positive
        return abs(c)
    else:
        # Clamp at zero
        return 0


def _jzs_integrand(g, r, n, p):
    return (
        (1 + g) ** ((n - p - 1) / 2)
        * (1 + (1 - r ** 2) * g) ** (-(n - 1) / 2)
        * g ** (-3 / 2)
        * math.exp(-n / (2 * g)))


def jzs_cor_bf(r, n):
    """
    Calculate the Bayes factor for a correlation value.

    Ported from R code in Wetzels & Wagenmakers (2012) "A default Bayesian hypothesis test for correlations and partial
    correlations". Psychon Bull Rev. 19:1057–1064 doi:10.3758/s13423-012-0295-x
    """

    bf10 = math.sqrt(n / 2) / math.gamma(1 / 2) * integrate.quad(func=_jzs_integrand, a=0, b=numpy.inf, args=(r, n, 1))[0]

    return bf10


def jzs_parcor_bf(r0, r1, n, p0, p1):
    """
    Calculate the Bayes factor for a partial correlation value.

    Ported from R code in Wetzels & Wagenmakers (2012) "A default Bayesian hypothesis test for correlations and partial
    correlations". Psychon Bull Rev. 19:1057–1064 doi:10.3758/s13423-012-0295-x
    """

    bf10 = (
        integrate.quad(func=_jzs_integrand, a=0, b=numpy.inf, args=(r1, n, p1))[0] /
        integrate.quad(func=_jzs_integrand, a=0, b=numpy.inf, args=(r0, n, p0))[0]
    )

    return bf10

