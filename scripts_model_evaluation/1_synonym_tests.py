"""
===========================
Evaluate using synonym tests.
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

from ..core.corpus.indexing import TokenIndexDictionary, FreqDist
from ..core.evaluation.synonym import ToeflTest, EslTest, LbmMcqTest, SynonymTester, SynonymResults
from ..core.model.count import PPMIModel, LogNgramModel, ConditionalProbabilityModel, ProbabilityRatioModel
from ..core.model.predict import SkipGramModel, CbowModel
from ..core.utils.maths import DistanceType
from ..core.utils.logging import log_message, date_format
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def main():

    test_battery = [
        ToeflTest(),
        EslTest(),
        LbmMcqTest()
    ]

    results = SynonymResults()
    results.load()

    for corpus_metadata in Preferences.source_corpus_metas:

        token_index = TokenIndexDictionary.load(corpus_metadata.index_path)
        freq_dist = FreqDist.load(corpus_metadata.freq_dist_path)

        for window_radius in Preferences.window_radii:

            # COUNT MODELS

            count_models = [
                LogNgramModel(corpus_metadata, window_radius, token_index),
                ConditionalProbabilityModel(corpus_metadata, window_radius, token_index, freq_dist),
                ProbabilityRatioModel(corpus_metadata, window_radius, token_index, freq_dist),
                PPMIModel(corpus_metadata, window_radius, token_index, freq_dist)
            ]

            for model in count_models:
                for test in test_battery:
                    for distance_type in DistanceType:
                        # TODO: horrifically inefficient: we load existing results each time
                        if not results.results_exist_for(test.name, model, distance_type):
                            model.train(memory_map=True)
                            results.extend_with_results(SynonymTester.administer_test(test, model, distance_type))
                            results.save()

                # release memory
                model.untrain()
            del count_models

            # PPMI (TRUNCATED, for replication of B&L 2007)
            model = PPMIModel(corpus_metadata, window_radius, token_index, freq_dist)
            truncate_length = 10_000
            for test in test_battery:
                for distance_type in DistanceType:
                    if not results.results_exist_for(test.name, model, distance_type, truncate_length):
                        model.train(memory_map=True)
                        results.extend_with_results(SynonymTester.administer_test(test, model, distance_type, truncate_length))
                        results.save()

            # release memory
            model.untrain()

            # PREDICT MODELS

            for embedding_size in Preferences.predict_embedding_sizes:

                predict_models = [
                    SkipGramModel(corpus_metadata, window_radius, embedding_size),
                    CbowModel(corpus_metadata, window_radius, embedding_size)
                ]

                for model in predict_models:
                    for test in test_battery:
                        for distance_type in DistanceType:
                            if not results.results_exist_for(test.name, model, distance_type, truncate_length):
                                model.train(memory_map=True)
                                results.extend_with_results(SynonymTester.administer_test(test, model, distance_type))
                                results.save()

                    # release memory
                    model.untrain()
                del predict_models


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
