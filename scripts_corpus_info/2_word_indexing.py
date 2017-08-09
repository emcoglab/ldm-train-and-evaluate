"""
===========================
Produce token–index dictionaries from a corpus, so tokens can easily be used to index vectors and matrices.
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

from ..core.corpus.distribution import FreqDist
from ..core.utils.indexing import TokenIndexDictionary
from ..core.utils.logging import log_message, date_format
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def main():
    for meta in Preferences.source_corpus_metas:
        logger.info(f"Producing word index dictionaries for {meta.name} corpus")

        freq_dist = FreqDist.load(meta.freq_dist_path)
        token_index = TokenIndexDictionary.from_freqdist(freq_dist)
        token_index.save(meta.index_path)


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
