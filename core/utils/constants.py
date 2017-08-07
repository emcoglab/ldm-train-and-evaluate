from enum import Enum


class Chirality(Enum):
    """
    Represents either left or right.
    """
    left = "left",
    right = "right"

    @property
    def name(self):
        if self == Chirality.left:
            return "left"
        elif self == Chirality.right:
            return "right"
        else:
            raise ValueError()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
