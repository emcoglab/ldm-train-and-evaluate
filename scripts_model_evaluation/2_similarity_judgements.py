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

from typing import List

from ..core.utils.maths import CorrelationType
from ..core.corpus.distribution import FreqDist
from ..core.evaluation.similarity import SimlexSimilarity, WordsimSimilarity, WordsimRelatedness, MenSimilarity, \
    SimilarityTester, SimilarityReportCard
from ..core.model.count import PPMIModel, LogNgramModel, ConditionalProbabilityModel, ProbabilityRatioModel
from ..core.model.predict import SkipGramModel, CbowModel
from ..core.utils.indexing import TokenIndexDictionary
from ..core.utils.logging import log_message, date_format
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def save_results(results: List[SimilarityReportCard.Entry]):
    csv_path = os.path.join(Preferences.similarity_judgement_results_dir, "similarity_judgements.csv")

    with open(csv_path, mode="w", encoding="utf-8") as csv_file:
        separator = ","
        for result in results:
            csv_file.write(separator.join(result.fields) + "\n")


def main():
    test_battery = [
        SimlexSimilarity(),
        WordsimSimilarity(),
        WordsimRelatedness(),
        MenSimilarity()
    ]

    correlation = CorrelationType.Pearson

    tester = SimilarityTester(test_battery)

    # TODO: this should skip, not overwrite, existing test results
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
                model.train()
                results = tester.administer_tests(model, correlation)
                save_results(results)

            # PREDICT MODELS

            for embedding_size in Preferences.predict_embedding_sizes:

                predict_models = [
                    SkipGramModel(corpus_metadata, window_radius, embedding_size),
                    CbowModel(corpus_metadata, window_radius, embedding_size)
                ]

                for model in predict_models:
                    model.train()
                    results = tester.administer_tests(model, correlation)
                    save_results(results)


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
