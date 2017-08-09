"""
===========================
Compute and save frequency distribution information from a corpus.
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
import sys

from ..core.utils.logging import log_message, date_format
from ..core.corpus.corpus import BatchedCorpus
from ..core.corpus.distribution import FreqDist
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def main():

    for corpus_meta in Preferences.source_corpus_metas:

        if FreqDist.could_load(corpus_meta.freq_dist_path):
            logger.info(f"Skipping ")
        else:

            logger.info(f"Loading corpus documents from {corpus_meta.path}")
            freq_dist = FreqDist.from_batched_corpus(BatchedCorpus(corpus_meta, batch_size=1_000_000))

            logger.info(f"Saving frequency distribution information to {corpus_meta.freq_dist_path}")
            freq_dist.save(corpus_meta.freq_dist_path)


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    main()

    logger.info("Done!")
