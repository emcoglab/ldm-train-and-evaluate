import logging

import nltk

from .corpus import BatchedCorpus

logger = logging.getLogger(__name__)


# noinspection PyAbstractClass
class FreqDistConstructor(nltk.probability.FreqDist):

    @classmethod
    def from_batched_corpus(cls, batched_corpus: BatchedCorpus) -> nltk.probability.FreqDist:
        freq_dist = nltk.probability.FreqDist()
        for batch in batched_corpus:
            freq_dist += nltk.probability.FreqDist(batch)
        return freq_dist
