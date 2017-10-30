"""
===========================
Removing low-frequency tokens from a tokenised corpus.
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

import logging
import os
import sys

from ..preferences.preferences import Preferences
from ..core.utils.logging import log_message, date_format
from ..core.corpus.corpus import CorpusMetadata, StreamedCorpus, BatchedCorpus
from ..core.corpus.indexing import FreqDist

logger = logging.getLogger(__name__)


def main():

    corpus_metas = [
        dict(
            source=Preferences.bbc_processing_metas["tokenised"],
            target=Preferences.bbc_processing_metas["filtered"]),
        dict(
            source=Preferences.bnc_processing_metas["tokenised"],
            target=Preferences.bnc_processing_metas["filtered"]),
        dict(
            source=Preferences.ukwac_processing_metas["tokenised"],
            target=Preferences.ukwac_processing_metas["filtered"])]

    # The frequency at which we ignore tokens.
    # Set to 0 to include all tokens, set to 1 to include tokens that occur more than once, etc.
    # TODO: decide if we want to do this or not
    ignorable_frequency = 1

    for corpus_meta in corpus_metas:

        freq_dist_path = os.path.join(corpus_meta["source"].freq_dist_path, corpus_meta["source"].name + ".corpus.pickle")
        if os.path.isfile(freq_dist_path):
            # If freq dist file previously saved, load it
            logger.info(f"Loading frequency distribution from {freq_dist_path}")
            freq_dist = FreqDist.load(freq_dist_path)
        else:
            # Compute it
            logger.info(f"Computing frequency distribution from {corpus_meta['source'].name} corpus")
            freq_dist = FreqDist.from_batched_corpus(
                BatchedCorpus(corpus_meta["source"], batch_size=1_000_000))

        logger.info(f"Loading {corpus_meta['source'].name} corpus from {corpus_meta['source'].path}")

        token_count = 0
        with open(corpus_meta["target"].path, mode="w", encoding="utf-8") as target_file:
            for token in StreamedCorpus(corpus_meta['source']):
                # Only write a token if it's sufficiently frequent
                if freq_dist[token] > ignorable_frequency:
                    target_file.write(token + "\n")

                    token_count += 1
                    if token_count % 1_000_000 == 0:
                        logging.info(f"\tWritten {token_count} tokens")


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    main()

    logger.info("Done!")
