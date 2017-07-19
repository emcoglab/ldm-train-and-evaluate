import argparse
import logging
import sys

import nltk

from ..core.tokenising import modified_word_tokenize
from ..core.filtering import filter_frequency

logger = logging.getLogger()


def main(corpus_dir, wordlist_dir):
    logger.info(f"Loading corpus' documents from {corpus_dir}")
    corpus = nltk.corpus.PlaintextCorpusReader(corpus_dir, ".+\..+")

    logger.info(f"Tokenising corpus")
    corpus = [w.lower() for w in modified_word_tokenize(corpus.raw())]
    vocab_corpus = set(corpus)

    logger.info(f"Loading wordlist from {wordlist_dir}")
    wordlist = nltk.corpus.PlaintextCorpusReader(wordlist_dir, ".+\..+")
    wordlist = [w.lower() for w in modified_word_tokenize(wordlist.raw())]
    vocab_wordlist = set(wordlist)

    logger.info(f"Corpus has a vocab of size {len(vocab_corpus)}")
    logger.info(f"Wordlist has a vocab of size {len(vocab_wordlist)}")

    frequency_dist = nltk.probability.FreqDist(corpus)

    for cutoff_freq in [0, 1, 5, 10, 50, 100, 500, 1000]:
        vocab_corpus = set([token
                            for token in filter_frequency(corpus,
                                                          ignore_tokens_with_frequencies_at_most=cutoff_freq,
                                                          freq_dist=frequency_dist)])
        logger.info(
            f"Overlap with cutoff freq {cutoff_freq} has a size of"
            f"\t{len(set.intersection(vocab_corpus, vocab_wordlist)):,}")
        logger.info(f"\tMissing words: {vocab_wordlist - vocab_corpus}")


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(module)s | %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    logger.info("")

    parser = argparse.ArgumentParser()
    parser.add_argument("corpus")
    parser.add_argument("wordlist")
    args = vars(parser.parse_args())

    main(corpus_dir=args["corpus"], wordlist_dir=args["wordlist"])

    logger.info("Done!")
