from enum import Enum


class Chiralities(Enum):
    """
    Represents either left or right.
    """
    left = "left",
    right = "right"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value
