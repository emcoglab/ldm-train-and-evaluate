"""
===========================
Evaluate LDMs using word association judgements.
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

from ldm.corpus.indexing import FreqDist
from ldm.evaluation.association import SimlexSimilarity, WordsimSimilarity, WordsimRelatedness, MenSimilarity, \
    AssociationTester, ThematicRelatedness, AssociationResults
from ldm.model.count import PPMIModel, LogCoOccurrenceCountModel, ConditionalProbabilityModel, ProbabilityRatioModel
from ldm.model.ngram import LogNgramModel, PPMINgramModel, ProbabilityRatioNgramModel
from ldm.model.predict import SkipGramModel, CbowModel
from ldm.preferences.preferences import Preferences
from ldm.utils.logging import log_message, date_format
from ldm.utils.maths import CorrelationType

from constants import DISTANCE_TYPES

logger = logging.getLogger(__name__)


def main():
    tester_battery = [
        AssociationTester(test=SimlexSimilarity()),
        AssociationTester(test=WordsimSimilarity()),
        AssociationTester(test=WordsimRelatedness()),
        AssociationTester(test=MenSimilarity()),
        AssociationTester(test=ThematicRelatedness()),
    ]

    results = AssociationResults()

    for corpus_metadata in Preferences.source_corpus_metas:

        freq_dist = FreqDist.load(corpus_metadata.freq_dist_path)

        for window_radius in Preferences.window_radii:

            ngram_models = [
                LogNgramModel(corpus_metadata, window_radius, freq_dist),
                PPMINgramModel(corpus_metadata, window_radius, freq_dist),
                ProbabilityRatioNgramModel(corpus_metadata, window_radius, freq_dist)
            ]

            for model in ngram_models:
                for tester in tester_battery:
                    if not tester.has_tested_model(model):
                        model.train(memory_map=True)
                        tester.administer_test(model)
                    else:
                        logger.info(f"Already run {tester.test.name} with {model.name}")
                    results.add_result(tester.test.name, model, None,
                                       tester.results_for_model(CorrelationType.Pearson, model))
                    results.add_result(tester.test.name, model, None,
                                       tester.results_for_model(CorrelationType.Spearman, model))
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
                for tester in tester_battery:
                    for distance_type in DISTANCE_TYPES:
                        if not tester.has_tested_model(model, distance_type):
                            model.train(memory_map=True)
                            tester.administer_test(model, distance_type)
                        else:
                            logger.info(f"Already run {tester.test.name} with {model.name}")
                        results.add_result(tester.test.name, model, distance_type,
                                           tester.results_for_model(CorrelationType.Pearson, model, distance_type))
                        results.add_result(tester.test.name, model, distance_type,
                                           tester.results_for_model(CorrelationType.Spearman, model, distance_type))
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
                    for tester in tester_battery:
                        for distance_type in DISTANCE_TYPES:
                            if not tester.has_tested_model(model, distance_type):
                                model.train(memory_map=True)
                                tester.administer_test(model, distance_type)
                            else:
                                logger.info(f"Already run {tester.test.name} with {model.name}")
                            results.add_result(tester.test.name, model, distance_type,
                                               tester.results_for_model(CorrelationType.Pearson, model, distance_type))
                            results.add_result(tester.test.name, model, distance_type,
                                               tester.results_for_model(CorrelationType.Spearman, model, distance_type))
                    # release memory
                    model.untrain()
                del predict_models

    results.save()


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
