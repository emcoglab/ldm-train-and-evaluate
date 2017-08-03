import logging

import nltk

from .corpus import BatchedCorpus

logger = logging.getLogger(__name__)


def freq_dist_from_corpus(metadata, batch_size=1_000_000, verbose=False):
    """
    Produces a nltk.probability.FreqDist from a corpus in batches, without loading the entire thing into RAM at one
    time.
    :param metadata:
    The filename of the corpus file. Should be a single text file with individual tokens on each line.
    :param batch_size:
    The number of tokens to load into RAM at one time.  Increasing will speed up function, but require more memory.
    :param verbose:
    Logging toggle
    :return freq_dist:
    A nltk.probability.FreqDist from the corpus.
    """

    # Read file directly, in batches, accumulating the FreqDist
    freq_dist = nltk.probability.FreqDist()
    batched_corpus = BatchedCorpus(metadata, batch_size=batch_size)

    logger.info("Building frequency distribution from corpus in batches")
    batch_i = 0
    for batch in batched_corpus:
        batch_i += 1
        if verbose:
            logger.info(f"\tWorking on batch {batch_i}, ({batch_i * batch_size} tokens)")
        freq_dist += nltk.probability.FreqDist(batch)

    return freq_dist
