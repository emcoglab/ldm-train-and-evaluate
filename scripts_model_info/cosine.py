"""
===========================
Query models using Euclidean distances.
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
from ..core.utils.maths import DistanceType
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def main():
    # TODO: Should work for all Preferences.window_radii
    window_radius = 1

    # TODO: Should work for all corpora
    corpus_metadata = Preferences.source_corpus_metas[0]  # BBC

    # TODO: Should work for vectors from all model types
    model = LogNgramModel(corpus_metadata,
                          Preferences.model_dir,
                          window_radius,
                          TokenIndexDictionary.load(corpus_metadata.index_path))

    model.train()

    w = "frog"
    n = 4
    d = DistanceType.cosine
    logger.info(f'Finding nearest {n} neighbours to "{w}"...')
    neighbours = model.nearest_neighbours(w, d, n)
    logger.info(f'Nearest {n} neighbours to "{w}": {neighbours}')


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
