import logging

import nltk


logger = logging.getLogger(__name__)


def freq_dist_from_file(filename, batch_size=1_000_000, verbose=False):
    """
    Produces a nltk.probability.FreqDist from a corpus in batches, without loading the entire thing into RAM at one
    time.
    :param filename:
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
    batch = []
    with open(filename, mode="r", encoding="utf-8") as corpus_file:

        # counters
        batch_count       = 0
        token_count       = 0
        tokens_this_batch = 0

        # Add tokens to the batch
        for line in corpus_file:
            batch.append(line.strip())
            tokens_this_batch += 1

            # When the batch is full
            if tokens_this_batch >= batch_size:
                # Dump the batch into the FreqDist
                freq_dist += nltk.probability.FreqDist(batch)

                batch_count += 1
                token_count += tokens_this_batch
                if verbose:
                    logger.info(f"{batch_count:,} batches processed, {token_count:,} tokens total")

                # Empty the batch
                batch = []
                tokens_this_batch = 0

        # Remember to do the final batch
        freq_dist += nltk.probability.FreqDist(batch)

    return freq_dist
