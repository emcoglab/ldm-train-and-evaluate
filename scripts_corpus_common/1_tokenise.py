"""
===========================
Tokenise corpora.
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
import sys
from os import path, mkdir

import nltk

from ..ldm.utils.logging import log_message, date_format
from ..ldm.corpus.filtering import filter_punctuation
from ..ldm.corpus.tokenising import modified_word_tokenize
from ..ldm.preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def main():

    # All corpora have been preprocessed and can now be tokenised in the same way
    corpus_metas = [
        dict(
            source=Preferences.bbc_processing_metas["replaced_symbols"],
            target=Preferences.bbc_processing_metas["tokenised"]),
        dict(
            source=Preferences.bnc_processing_metas["detagged"],
            target=Preferences.bnc_processing_metas["tokenised"]),
        dict(
            source=Preferences.bnc_text_processing_metas["detagged"],
            target=Preferences.bnc_text_processing_metas["tokenised"]),
        dict(
            source=Preferences.bnc_speech_processing_metas["detagged"],
            target=Preferences.bnc_speech_processing_metas["tokenised"]),
        dict(
            source=Preferences.ukwac_processing_metas["partitioned"],
            target=Preferences.ukwac_processing_metas["tokenised"])]

    token_delimiter = "\n"

    for corpus_meta in corpus_metas:

        token_count = 0

        # Skip corpora which are already done.
        if path.isfile(corpus_meta['target'].path):
            logger.info(f"The file {path.basename(corpus_meta['target'].path)} already exists in this location.")
            logger.info(f"Skipping!")
            continue

        source_paths = glob.glob(path.join(corpus_meta['source'].path, "*.*"))
        # The should be loaded in the order that they were produced
        source_doc_filenames = sorted([path.basename(p) for p in source_paths])

        logger.info(f"Loading {corpus_meta['source'].name} corpus from {corpus_meta['source'].path}")

        for source_doc_filename in source_doc_filenames:

            corpus_doc = nltk.corpus.PlaintextCorpusReader(corpus_meta['source'].path, source_doc_filename).raw()
            corpus_doc = modified_word_tokenize(corpus_doc)

            # Filter punctuation
            corpus_doc = filter_punctuation(corpus_doc)

            # Tokens are case-insensitive
            corpus_doc = [t.lower() for t in corpus_doc]

            if not path.isdir(path.dirname(corpus_meta['target'].path)):
                logger.warning(f"{corpus_meta['target'].path} doesn't exist, making it.")
                mkdir(path.dirname(corpus_meta['target'].path))

            with open(corpus_meta['target'].path, mode="a", encoding="utf-8") as tokenised_corpus_file:

                for token in corpus_doc:

                    tokenised_corpus_file.write(
                        token + token_delimiter)
                    token_count += 1

                    if token_count % 100_000 == 0 and token_count > 0:
                        logger.info(f"\tWritten {token_count:,} tokens")


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))

    main()

    logger.info("Done!")
