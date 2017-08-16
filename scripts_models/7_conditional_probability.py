"""
===========================
Conditional probability model.
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
from ..core.model.count import ConditionalProbabilityModel
from ..core.utils.indexing import TokenIndexDictionary
from ..preferences.preferences import Preferences
from ..core.utils.logging import log_message, date_format

logger = logging.getLogger(__name__)


def main():

    for meta in Preferences.source_corpus_metas:
        token_indices = TokenIndexDictionary.load(meta.index_path)
        freq_dist = FreqDist.load(meta.freq_dist_path)
        for radius in Preferences.window_radii:
            model = ConditionalProbabilityModel(meta, Preferences.model_dir, radius, token_indices, freq_dist)
            if not model.could_load:
                model.train()


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
