"""
===========================
(Summed) n-gram counts.
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

from ..ldm.corpus.indexing import FreqDist
from ..ldm.model.count import CoOccurrenceCountModel
from ..ldm.preferences.preferences import Preferences

logger = logging.getLogger()


def main():

    for corpus_meta in Preferences.source_corpus_metas:
        freq_dist = FreqDist.load(corpus_meta.freq_dist_path)
        for radius in Preferences.window_radii:
            model = CoOccurrenceCountModel(corpus_meta, radius, freq_dist)
            if not model.could_load:
                model.train()


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
