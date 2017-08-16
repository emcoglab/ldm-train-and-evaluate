"""
===========================
Evaluate using TOEFL test.
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
from ..core.model.count import PPMIModel, LogNgramModel, ConditionalProbabilityModel, ProbabilityRatioModel
from ..core.model.evaluation import ToeflTest, SynonymTester
from ..core.utils.indexing import TokenIndexDictionary
from ..core.utils.logging import log_message, date_format
from ..core.utils.maths import DistanceType
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def main():

    test = ToeflTest()

    for corpus_metadata in Preferences.source_corpus_metas:

        token_index = TokenIndexDictionary.load(corpus_metadata.index_path)
        freq_dist = FreqDist.load(corpus_metadata.freq_dist_path)

        for window_radius in Preferences.window_radii:

                # Run through each model

                # Log n-gram
                model = LogNgramModel(
                    corpus_metadata, Preferences.model_dir, window_radius, token_index)
                for distance_type in DistanceType:
                    tester = SynonymTester(model, test, distance_type)
                    # Skip ones we've done
                    if tester.saved_transcript_exists:
                        continue
                    if not tester.model.is_trained:
                        tester.model.train()

                # Log n-gram
                model = LogNgramModel(
                    corpus_metadata, Preferences.model_dir, window_radius, token_index)
                for distance_type in DistanceType:
                    tester = SynonymTester(model, test, distance_type)
                    # Skip ones we've done
                    if tester.saved_transcript_exists:
                        continue
                    if not tester.model.is_trained:
                        tester.model.train()

                # Conditional probability
                model = ConditionalProbabilityModel(
                    corpus_metadata, Preferences.model_dir, window_radius, token_index, freq_dist)
                for distance_type in DistanceType:
                    tester = SynonymTester(model, test, distance_type)
                    # Skip ones we've done
                    if tester.saved_transcript_exists:
                        continue
                    if not tester.model.is_trained:
                        tester.model.train()

                # Probability ratios
                model = ProbabilityRatioModel(
                    corpus_metadata, Preferences.model_dir, window_radius, token_index, freq_dist)
                for distance_type in DistanceType:
                    tester = SynonymTester(model, test, distance_type)
                    # Skip ones we've done
                    if tester.saved_transcript_exists:
                        continue
                    if not tester.model.is_trained:
                        tester.model.train()

                # PPMI
                model = PPMIModel(
                    corpus_metadata, Preferences.model_dir, window_radius, token_index, freq_dist)
                for distance_type in DistanceType:
                    tester = SynonymTester(model, test, distance_type)
                    # Skip ones we've done
                    if tester.saved_transcript_exists:
                        continue
                    if not tester.model.is_trained:
                        tester.model.train()


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
