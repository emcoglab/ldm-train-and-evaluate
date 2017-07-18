import argparse
import logging
import sys

import nltk

from ..core.tokenising import modified_word_tokenize
from ..core.filtering import filter_frequency

logger = logging.getLogger()


def main(corpus_dir_1, corpus_dir_2):
    logger.info(f"Loading 1st corpus' documents from {corpus_dir_1}")
    corpus_1 = nltk.corpus.PlaintextCorpusReader(corpus_dir_1, ".+\..+")
    logger.info(f"Tokenising corpus")
    corpus_1 = [w.lower() for w in modified_word_tokenize(corpus_1.raw())]
    vocab_1 = set(corpus_1)

    logger.info(f"Loading 2nd corpus' documents from {corpus_dir_2}")
    corpus_2 = nltk.corpus.PlaintextCorpusReader(corpus_dir_2, ".+\..+")
    logger.info(f"Tokenising corpus")
    corpus_2 = [w.lower() for w in modified_word_tokenize(corpus_2.raw())]
    vocab_2 = set(corpus_2)

    logger.info(f"Corpus 1 has a vocab of size {len(vocab_1)}")
    logger.info(f"Corpus 2 has a vocab of size {len(vocab_2)}")

    overlap = set.intersection(vocab_1, vocab_2)

    logger.info(f"Overlap has a size of {len(overlap)}")

    logger.info("Checking overlap with frequency-filtered 1st corpus")

    frequency_dist_1 = nltk.probability.FreqDist(corpus_1)

    for cutoff_freq in [1, 5, 10, 50, 100, 500, 1000]:
        vocab_1 = set([token for token in filter_frequency(corpus_1, min_freq=cutoff_freq+1, freq_dist=frequency_dist_1)])
        overlap = set.intersection(vocab_1, vocab_2)
        logger.info(f"Overlap with cutoff freq {cutoff_freq} has a size of {len(overlap)}")



if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(module)s | %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    logger.info("")

    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_dir_1")
    parser.add_argument("corpus_dir_2")
    args = vars(parser.parse_args())

    main(corpus_dir_1=args["corpus_dir_1"], corpus_dir_2=args["corpus_dir_2"])

    logger.info("Done!")
