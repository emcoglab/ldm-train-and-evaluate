from enum import Enum

import numpy
from scipy import spatial


class DistanceType(Enum):
    """
    Representative of a distance type.
    """
    Euclidean = 0
    cosine = 1


def distance(u: numpy.ndarray, v: numpy.ndarray, distance_type: DistanceType) -> float:
    """
    Distance from vector u to vector v using the specified distance type.
    """

    if distance_type is DistanceType.Euclidean:
        return _euclidean(u, v)
    elif distance_type is DistanceType.cosine:
        return _cosine(u, v)
    else:
        raise ValueError()


def _euclidean(u: numpy.ndarray, v: numpy.ndarray):
    """
    Euclidean distance.
    :param u:
    :param v:
    :return:
    """
    return numpy.linalg.norm(u - v)


def _cosine(u: numpy.ndarray, v: numpy.ndarray):
    """
    Cosine distance.
    :param u:
    :param v:
    :return:
    """
    return spatial.distance.cosine(u, v)
