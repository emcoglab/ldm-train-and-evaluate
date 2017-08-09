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
    Extension of nltk.probability.FreqDist with a few useful helper methods.
    """

    def save(self, filename):
        with open(filename, mode="wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename) -> 'FreqDist':
        with open(filename, mode="rb") as file:
            return pickle.load(file)

    @classmethod
    def from_batched_corpus(cls, batched_corpus: BatchedCorpus) -> 'FreqDist':
        # Initially create an empty FreqDist
        freq_dist = cls()
        # Then add to it from each batch of the corpus.
        # Means we don't have to have the whole corpus in memory at once!
        token_count = 0
        for batch in batched_corpus:
            freq_dist += cls(batch)
            token_count += batched_corpus.batch_size
            logger.info(f"\t{token_count:,} tokens counted")
        return freq_dist
