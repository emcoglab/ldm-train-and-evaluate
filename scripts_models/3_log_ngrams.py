"""
===========================
Log n-gram model.
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

from ..core.model.count import LogNgramModel
from ..core.utils.indexing import TokenIndexDictionary
from ..preferences.preferences import Preferences
from ..core.utils.logging import log_message, date_format

logger = logging.getLogger(__name__)


def main():

    for meta in Preferences.source_corpus_metas:
        token_indices = TokenIndexDictionary.load(meta.index_path)
        for radius in Preferences.window_radii:
            model = LogNgramModel(meta, Preferences.model_dir, radius, token_indices)
            if not model.could_load:
                model.train()


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
