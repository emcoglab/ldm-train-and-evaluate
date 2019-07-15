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
import sys

from constants import DISTANCE_TYPES
from ldm.corpus.indexing import FreqDist
from ldm.evaluation.association import SimlexSimilarity, WordsimSimilarity, WordsimRelatedness, MenSimilarity, \
    AssociationTester, ColourEmotionAssociation, ThematicRelatedness, AssociationResults
from ldm.model.count import PPMIModel, LogCoOccurrenceCountModel, ConditionalProbabilityModel, ProbabilityRatioModel
from ldm.model.ngram import LogNgramModel, PPMINgramModel, ProbabilityRatioNgramModel
from ldm.model.predict import SkipGramModel, CbowModel
from ldm.preferences.preferences import Preferences
from ldm.utils.logging import log_message, date_format

logger = logging.getLogger(__name__)


def main():
    test_battery = [
        SimlexSimilarity(),
        WordsimSimilarity(),
        WordsimRelatedness(),
        MenSimilarity(),
        ColourEmotionAssociation(),
        ThematicRelatedness()
        # ThematicRelatedness(only_use_response=1)
    ]

    results = AssociationResults()
    results.load()

    for corpus_metadata in Preferences.source_corpus_metas:

        freq_dist = FreqDist.load(corpus_metadata.freq_dist_path)

        for window_radius in Preferences.window_radii:

            ngram_models = [
                LogNgramModel(corpus_metadata, window_radius, freq_dist),
                PPMINgramModel(corpus_metadata, window_radius, freq_dist),
                ProbabilityRatioNgramModel(corpus_metadata, window_radius, freq_dist)
            ]

            for model in ngram_models:
                for test in test_battery:
                    if not results.results_exist_for(test.name, model, None):
                        model.train(memory_map=True)
                        results.extend_with_results(AssociationTester.administer_test(test, model, None))
                        results.save()
                # release memory
                model.untrain()
            del ngram_models

            # COUNT MODELS

            count_models = [
                LogCoOccurrenceCountModel(corpus_metadata, window_radius, freq_dist),
                ConditionalProbabilityModel(corpus_metadata, window_radius, freq_dist),
                ProbabilityRatioModel(corpus_metadata, window_radius, freq_dist),
                PPMIModel(corpus_metadata, window_radius, freq_dist)
            ]

            for model in count_models:
                for test in test_battery:
                    for distance_type in DISTANCE_TYPES:
                        if not results.results_exist_for(test.name, model, distance_type):
                            model.train(memory_map=True)
                            results.extend_with_results(AssociationTester.administer_test(test, model, distance_type))
                            results.save()
                # release memory
                model.untrain()
            del count_models

            # PREDICT MODELS

            for embedding_size in Preferences.predict_embedding_sizes:

                predict_models = [
                    SkipGramModel(corpus_metadata, window_radius, embedding_size),
                    CbowModel(corpus_metadata, window_radius, embedding_size)
                ]

                for model in predict_models:
                    for test in test_battery:
                        for distance_type in DISTANCE_TYPES:
                            if not results.results_exist_for(test.name, model, distance_type):
                                model.train(memory_map=True)
                                results.extend_with_results(
                                    AssociationTester.administer_test(test, model, distance_type))
                                results.save()

                    # release memory
                    model.untrain()
                del predict_models


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
