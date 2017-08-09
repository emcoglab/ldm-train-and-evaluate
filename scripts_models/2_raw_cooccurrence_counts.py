"""
===========================
Raw cooccurrence counts at various fixed distances.
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

from ..core.utils.constants import Chirality
from ..core.model.count import UnsummedNgramCountModel
from ..core.utils.indexing import TokenIndexDictionary
from ..preferences.preferences import Preferences

logger = logging.getLogger()


def main():
    for meta in Preferences.source_corpus_metas:
        token_indices = TokenIndexDictionary.load(meta.index_path)
        for radius in range(1, max(Preferences.window_radii) + 1):
            for chirality in Chirality:
                model = UnsummedNgramCountModel(meta, Preferences.model_dir, radius, token_indices, chirality)
                model.train(load_if_previously_saved=False)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
