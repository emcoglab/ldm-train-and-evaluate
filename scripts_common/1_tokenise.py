import logging
import sys

import nltk

from ..core.classes import SourceTargetPair, CorpusMetaData
from ..core.filtering import filter_punctuation, ignorable_punctuation
from ..core.tokenising import modified_word_tokenize

logger = logging.getLogger()


def main():

    corpus_metas = [
        SourceTargetPair(
            source=CorpusMetaData(
                name="BBC",  path="/Users/caiwingfield/corpora/BBC/3 Replaced symbols"),
            target=CorpusMetaData(
                name="BBC", path="/Users/caiwingfield/corpora/BBC/4 Tokenised/BBC.corpus")),
        SourceTargetPair(
            source=CorpusMetaData(
                name="BNC", path="/Users/caiwingfield/corpora/BNC/1 Detagged"),
            target=CorpusMetaData(
                name="BNC",  path="/Users/caiwingfield/corpora/BNC/2 Tokenised/BNC.corpus")),
        SourceTargetPair(
            source=CorpusMetaData(
                name="UKWAC", path="/Users/caiwingfield/corpora/UKWAC/1 Text only"),
            target=CorpusMetaData(
                name="UKWAC", path="/Users/caiwingfield/corpora/UKWAC/2 Tokenised/UKWAC.corpus"))]

    token_delimiter = "\n"

    for corpus_meta in corpus_metas:
        logger.info(f"Loading {corpus_meta.source.name} corpus from {corpus_meta.source.path}")
        logger.info(f"Tokenising corpus")
        corpus = modified_word_tokenize(
            # Any file with name and extension
            nltk.corpus.PlaintextCorpusReader(corpus_meta.source.path, ".+\..+").raw())

        corpus_size = len(corpus)
        logger.info(f"{corpus_size:,} tokens in corpus")

        # Filter punctuation
        logger.info(f"Filtering punctuation out of corpus: {ignorable_punctuation}")
        corpus = filter_punctuation(corpus)

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
