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

from ..core.utils.logging import log_message, date_format
from ..core.corpus.filtering import filter_punctuation
from ..core.corpus.tokenising import modified_word_tokenize
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def main():

    corpus_metas = [
        dict(
            source=Preferences.bbc_processing_metas["replaced_symbols"],
            target=Preferences.bnc_processing_metas["tokenised"]),
        dict(
            source=Preferences.bnc_processing_metas["detagged"],
            target=Preferences.bnc_processing_metas["tokenised"]),
        dict(
            source=Preferences.ukwac_processing_metas["partitioned"],
            target=Preferences.ukwac_processing_metas["tokenised"])]

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
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))

    main()

    logger.info("Done!")
