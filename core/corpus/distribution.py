"""
===========================
Classes related to frequency distribution.
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

import logging
import pickle

import nltk

from .corpus import BatchedCorpus

logger = logging.getLogger(__name__)


# noinspection PyAbstractClass
class FreqDist(nltk.probability.FreqDist):
    """
    Extension methods for the nltk.probability.FreqDist class
    """

    def save(self, filename):
        with open(filename, mode="wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename) -> nltk.probability.FreqDist:
        with open(filename, mode="rb") as file:
            return pickle.load(file)

    @classmethod
    def from_batched_corpus(cls, batched_corpus: BatchedCorpus) -> nltk.probability.FreqDist:
        freq_dist = nltk.probability.FreqDist()
        for batch in batched_corpus:
            freq_dist += nltk.probability.FreqDist(batch)
        return freq_dist
