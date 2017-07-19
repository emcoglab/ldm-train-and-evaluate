import os
import sys
import logging

import nltk

from ..core.filtering import filter_frequency, filter_punctuation, ignorable_punctuation
from ..core.tokenising import modified_word_tokenize

logger = logging.getLogger()


class SourceTargetPair:
    """
    A pair of a source directory and a target directory.
    """
    def __init__(self, source, target):
        self.source = source
        self.target = target


def main():

    corpus_dirs = [
        SourceTargetPair(
            "/Users/caiwingfield/Langboot local/Corpora/Combined/0 SPEECH",
            "/Users/caiwingfield/Langboot local/Corpora/Combined/1 SPEECH tokenised filtered"),
        SourceTargetPair(
            "/Users/caiwingfield/Langboot local/Corpora/Combined/0 TEXT",
            "/Users/caiwingfield/Langboot local/Corpora/Combined/1 TEXT tokenised filtered")]

    # The frequency at which we ignore tokens.
    # Set to 0 to include all tokens, set to 1 to include tokens that occur more than once, etc.
    ignorable_frequency = 1

    for corpus_dir in corpus_dirs:
        logger.info(f"Loading corpus from {corpus_dir.source}")
        logger.info(f"Tokenising corpus")
        corpus = modified_word_tokenize(
            # Any file with name and extension
            nltk.corpus.PlaintextCorpusReader(corpus_dir.source, ".+\..+").raw())

        corpus_size = len(corpus)
        logger.info(f"{corpus_size:,} tokens in corpus")

        # Filter punctuation
        logger.info(f"Filtering punctuation out of corpus: {ignorable_punctuation}")
        corpus = filter_punctuation(corpus)

        corpus_size = len(corpus)
        logger.info(f"{corpus_size:,} tokens in corpus")

        if ignorable_frequency > 0:
            logger.info(f"Filtering corpus based on token frequency")
            logger.info(f"Removing all tokens appearing fewer than {ignorable_frequency} times")
            corpus = filter_frequency(corpus, ignore_tokens_with_frequencies_at_most=ignorable_frequency)

            corpus_size = len(corpus)
            logger.info(f"{corpus_size:,} tokens in corpus")

        corpus_filename = os.path.join(corpus_dir.target, "corpus.txt")
        logger.info(f"Saving corpus to {corpus_filename}")
        with open(corpus_filename, mode="w", encoding="utf-8") as corpus_file:
            for i, token in enumerate(corpus):
                corpus_file.write(token+"\n")
                if i % 1_000_000 == 0 and i > 0:
                    logger.info(f"\tWritten {i:,}/{corpus_size:,} tokens ({100*(i/corpus_size):.0}%)")


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))

    main()

    logger.info("Done!")
