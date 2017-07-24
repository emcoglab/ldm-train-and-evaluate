import logging
import sys

import nltk

from ..core.classes import SourceTargetPair, CorpusMetaData
from ..core.filtering import filter_frequency

logger = logging.getLogger()


def main():

    corpus_metas = [
        # TODO: these paths are wrong. Fix and rerun
        SourceTargetPair(
            source=CorpusMetaData(
                name="BBC",  path="/Users/caiwingfield/corpora/BBC/4 Tokenised/BBC.corpus"),
            target=CorpusMetaData(
                name="BBC", path="/Users/caiwingfield/corpora/BBC/4 Tokenised/BBC.corpus")),
        SourceTargetPair(
            source=CorpusMetaData(
                name="BNC", path="/Users/caiwingfield/corpora/BNC/2 Tokenised/BNC.corpus"),
            target=CorpusMetaData(
                name="BNC",  path="/Users/caiwingfield/corpora/BNC/2 Tokenised/BNC.corpus"))]

    # The frequency at which we ignore tokens.
    # Set to 0 to include all tokens, set to 1 to include tokens that occur more than once, etc.
    ignorable_frequency = 1

    token_delimiter = "\n"

    for corpus_meta in corpus_metas:
        logger.info(f"Loading {corpus_meta.source.name} corpus from {corpus_meta.source.path}")

        logger.info(f"Tokenising corpus")
        # TODO: Just open the files, the tokenisation is done. CorpusReader is too slow
        corpus = nltk.corpus.PlaintextCorpusReader(corpus_meta.source.path, ".+\..+").raw().split("\n")

        corpus_size = len(corpus)
        logger.info(f"{corpus_size:,} tokens in corpus")

        if ignorable_frequency > 0:
            logger.info(f"Filtering corpus based on token frequency")
            logger.info(f"Removing all tokens appearing at most {ignorable_frequency} times")
            corpus = filter_frequency(corpus, ignore_tokens_with_frequencies_at_most=ignorable_frequency)

            corpus_size = len(corpus)
            logger.info(f"{corpus_size:,} tokens in corpus")

        logger.info(f"Saving {corpus_meta.target.name} corpus to {corpus_meta.target.path}")
        with open(corpus_meta.target.path, mode="w", encoding="utf-8") as corpus_file:
            for i, token in enumerate(corpus):
                corpus_file.write(token+token_delimiter)
                if i % 1_000_000 == 0 and i > 0:
                    logger.info(f"\tWritten {i:,}/{corpus_size:,} tokens ({int(100*(i/corpus_size))}%)")


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))

    main()

    logger.info("Done!")
