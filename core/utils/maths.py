import math
from enum import Enum

import numpy as np


class Distance(object):
    class Type(Enum):
        """
        Representative of a distance type
        """
        Euclidean = 0
        cosine = 1

    @staticmethod
    def d(u, v, distance_type: Type) -> float:
        """
        Distance from vector u to vector v
        :param u:
        :param v:
        :param distance_type:
        :return:
        """
        if distance_type is Distance.Type.Euclidean:
            return math.sqrt(sum([e ** 2 for e in u - v]))
        elif distance_type is Distance.Type.cosine:
            # TODO: What is the fast way to do this again?
            return np.dot(u, v) / 0
        else:
            raise ValueError()
