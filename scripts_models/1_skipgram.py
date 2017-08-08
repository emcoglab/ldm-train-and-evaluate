"""
===========================
Skip-gram model.
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

from ..core.model.predict import SkipGramModel
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def main():

    weights_dir = "/Users/caiwingfield/vectors/"

    for meta in Preferences.source_corpus_metas:
        for embedding_size in Preferences.predict_embedding_sizes:
            for window_radius in Preferences.window_radii:

                predict_model = SkipGramModel(meta, weights_dir, window_radius, embedding_size)
                predict_model.train()
                predict_model.save()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
