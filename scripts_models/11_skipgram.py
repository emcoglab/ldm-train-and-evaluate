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

from ..ldm.model.predict import SkipGramModel
from ..ldm.preferences.preferences import Preferences
from ..ldm.utils.logging import log_message, date_format

logger = logging.getLogger(__name__)


def main_parallel(n_workers: int):
    for corpus_meta in Preferences.source_corpus_metas:
        for embedding_size in Preferences.predict_embedding_sizes:
            for window_radius in Preferences.window_radii:
                predict_model = SkipGramModel(corpus_meta, window_radius, embedding_size, n_workers=n_workers)
                if not predict_model.could_load:
                    predict_model.train()


def main():
    for corpus_meta in Preferences.source_corpus_metas:
        for embedding_size in Preferences.predict_embedding_sizes:
            for window_radius in Preferences.window_radii:
                predict_model = SkipGramModel(corpus_meta, window_radius, embedding_size)
                if not predict_model.could_load:
                    predict_model.train()


if __name__ == '__main__':
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
