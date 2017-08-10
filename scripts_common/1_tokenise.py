"""
===========================
Tokenisation of a corpus of natural language.
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

import glob
import logging
import os
import sys

import nltk

from ..core.corpus.tokenising import modified_word_tokenize
from ..core.corpus.corpus import CorpusMetadata
from ..core.corpus.filtering import filter_punctuation

logger = logging.getLogger()


def main():

    corpus_metas = [
        dict(
            source=CorpusMetadata(
                name="BBC",  path="/Users/caiwingfield/corpora/BBC/3 Replaced symbols"),
            target=CorpusMetadata(
                name="BBC", path="/Users/caiwingfield/corpora/BBC/4 Tokenised/BBC.corpus")),
        dict(
            source=CorpusMetadata(
                name="BNC", path="/Users/caiwingfield/corpora/BNC/1 Detagged"),
            target=CorpusMetadata(
                name="BNC",  path="/Users/caiwingfield/corpora/BNC/2 Tokenised/BNC.corpus")),
        dict(
            source=CorpusMetadata(
                name="UKWAC", path="/Users/caiwingfield/corpora/UKWAC/2 Partitioned"),
            target=CorpusMetadata(
                name="UKWAC", path="/Users/caiwingfield/corpora/UKWAC/3 Tokenised/UKWAC.corpus"))]

    token_delimiter = "\n"

    for corpus_meta in corpus_metas:

        token_count = 0

        # Skip corpora which are already done.
        if os.path.isfile(corpus_meta['target'].path):
            logger.info(f"The file {os.path.basename(corpus_meta['target'].path)} already exists in this location.")
            logger.info(f"Skipping!")
            continue

        source_paths = glob.glob(os.path.join(corpus_meta['source'].path, "*.*"))
        # The should be loaded in the order that they were produced
        source_filenames = sorted([os.path.basename(path) for path in source_paths])

        logger.info(f"Loading {corpus_meta['source'].name} corpus from {corpus_meta['source'].path}")

        for source_filename in source_filenames:

            corpus = nltk.corpus.PlaintextCorpusReader(corpus_meta['source'].path, source_filename).raw()
            corpus = modified_word_tokenize(corpus)

            # Filter punctuation
            corpus = filter_punctuation(corpus)

            with open(corpus_meta['target'].path, mode="a", encoding="utf-8") as tokenised_corpus_file:

                for token in corpus:

                    tokenised_corpus_file.write(
                        # Tokens are case-insensitive
                        token.lower()
                        + token_delimiter)
                    token_count += 1

                    if token_count % 100_000 == 0 and token_count > 0:
                        logger.info(f"\tWritten {token_count:,} tokens")


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))

    main()

    logger.info("Done!")
