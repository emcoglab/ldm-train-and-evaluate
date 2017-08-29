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

from ..core.corpus.distribution import FreqDist
from ..core.evaluation.similarity import SimlexSimilarity, WordsimSimilarity, WordsimRelatedness, SimilarityTester, \
    SimilarityTestResult
from ..core.model.count import PPMIModel, LogNgramModel, ConditionalProbabilityModel, ProbabilityRatioModel
from ..core.model.predict import SkipGramModel, CbowModel
from ..core.utils.indexing import TokenIndexDictionary
from ..core.utils.logging import log_message, date_format
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def save_results(results: List[SimilarityTestResult]):
    results_dir = os.path.join(Preferences.eval_dir, "similarity")
    csv_name = os.path.join(results_dir, "similarity_judgements.csv")

    separator = ","

    with open(csv_name, mode="a", encoding="utf-8") as csv_file:
        for result in results:
            csv_file.write(separator.join(result.fields) + "\n")


def main():
    test_battery = [SimlexSimilarity(), WordsimSimilarity(), WordsimRelatedness()]

    # TODO: this should skip, not overwrite, existing test results
    for corpus_metadata in Preferences.source_corpus_metas:

        token_index = TokenIndexDictionary.load(corpus_metadata.index_path)
        freq_dist = FreqDist.load(corpus_metadata.freq_dist_path)

        tester = SimilarityTester(test_battery)

        for window_radius in Preferences.window_radii:

            # COUNT MODELS

            # Log n-gram
            model = LogNgramModel(corpus_metadata, Preferences.model_dir, window_radius, token_index)
            model.train()
            results = tester.administer_tests(model)
            save_results(results)

            # Conditional probability
            model = ConditionalProbabilityModel(corpus_metadata, Preferences.model_dir, window_radius, token_index,
                                                freq_dist)
            model.train()
            results = tester.administer_tests(model)
            save_results(results)

            # Probability ratios
            model = ProbabilityRatioModel(corpus_metadata, Preferences.model_dir, window_radius, token_index,
                                          freq_dist)
            model.train()
            results = tester.administer_tests(model)
            save_results(results)

            # PPMI
            model = PPMIModel(corpus_metadata, Preferences.model_dir, window_radius, token_index, freq_dist)
            model.train()
            results = tester.administer_tests(model)
            save_results(results)

            # PREDICT MODELS

            for embedding_size in Preferences.predict_embedding_sizes:
                # Skip-gram
                model = SkipGramModel(corpus_metadata, Preferences.model_dir, window_radius, embedding_size)
                model.train()
                results = tester.administer_tests(model)
                save_results(results)

                # CBOW
                model = CbowModel(corpus_metadata, Preferences.model_dir, window_radius, embedding_size)
                model.train()
                results = tester.administer_tests(model)
                save_results(results)


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
