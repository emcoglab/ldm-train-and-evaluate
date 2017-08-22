"""
===========================
Evaluate using word similarity judgements.
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

from ..core.corpus.distribution import FreqDist
from ..core.evaluation.similarity import Simlex, WordsimSimilarity, WordsimRelatedness, SimilarityTester
from ..core.model.count import PPMIModel, LogNgramModel, ConditionalProbabilityModel, ProbabilityRatioModel
from ..core.model.predict import SkipGramModel, CbowModel
from ..core.utils.indexing import TokenIndexDictionary
from ..core.utils.logging import log_message, date_format
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def main():
    report_card_dir = os.path.join(Preferences.eval_dir, "report cards")
    csv_name_pattern = "{model_name}.csv"

    test_battery = [Simlex(), WordsimSimilarity(), WordsimRelatedness()]

    for corpus_metadata in Preferences.source_corpus_metas:

        token_index = TokenIndexDictionary.load(corpus_metadata.index_path)
        freq_dist = FreqDist.load(corpus_metadata.freq_dist_path)

        tester = SimilarityTester(test_battery)

        for window_radius in Preferences.window_radii:

            # COUNT MODELS

            # PPMI
            model = PPMIModel(corpus_metadata, Preferences.model_dir, window_radius, token_index, freq_dist)
            model.train()
            tester.administer_tests(model)


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
