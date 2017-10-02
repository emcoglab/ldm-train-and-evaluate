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
from ..core.evaluation.synonym import ToeflTest, EslTest, McqTest, SynonymTester, ReportCard
from ..core.model.count import PPMIModel, LogNgramModel, ConditionalProbabilityModel, ProbabilityRatioModel
from ..core.model.predict import SkipGramModel, CbowModel
from ..core.utils.indexing import TokenIndexDictionary
from ..core.utils.logging import log_message, date_format
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def main():

    test_battery = [
        ToeflTest(),
        EslTest(),
        McqTest()
    ]

    for corpus_metadata in Preferences.source_corpus_metas:

        token_index = TokenIndexDictionary.load(corpus_metadata.index_path)
        freq_dist = FreqDist.load(corpus_metadata.freq_dist_path)

        for window_radius in Preferences.window_radii:

            # COUNT MODELS

            # Log n-gram
            model = LogNgramModel(corpus_metadata, window_radius, token_index)
            csv_name = model.name + '.csv'
            if not ReportCard.saved_with_name(csv_name):
                model.train()
                report_card = SynonymTester.administer_tests(model, test_battery)
                report_card.save_csv(csv_name)

            # Conditional probability
            model = ConditionalProbabilityModel(corpus_metadata, window_radius, token_index, freq_dist)
            csv_name = model.name + '.csv'
            if not ReportCard.saved_with_name(csv_name):
                model.train()
                report_card = SynonymTester.administer_tests(model, test_battery)
                report_card.save_csv(csv_name)

            # Probability ratios
            model = ProbabilityRatioModel(corpus_metadata, window_radius, token_index, freq_dist)
            csv_name = model.name + '.csv'
            if not ReportCard.saved_with_name(csv_name):
                model.train()
                report_card = SynonymTester.administer_tests(model, test_battery)
                report_card.save_csv(csv_name)

            # PPMI
            model = PPMIModel(corpus_metadata, window_radius, token_index, freq_dist)
            csv_name = model.name + '.csv'
            if not ReportCard.saved_with_name(csv_name):
                model.train()
                report_card = SynonymTester.administer_tests(model, test_battery)
                report_card.save_csv(csv_name)

            # PPMI (TRUNCATED, for replication of B&L 2007)
            truncate_length = 10_000
            csv_name = model.name + ' (10k).csv'
            if not ReportCard.saved_with_name(csv_name):
                model.train()
                report_card = SynonymTester.administer_tests(model, test_battery, truncate_length)
                report_card.save_csv(csv_name)

            # PREDICT MODELS

            for embedding_size in Preferences.predict_embedding_sizes:

                # Skip-gram
                model = SkipGramModel(corpus_metadata, window_radius, embedding_size)
                csv_name = model.name + '.csv'
                if not ReportCard.saved_with_name(csv_name):
                    model.train()
                    report_card = SynonymTester.administer_tests(model, test_battery)
                    report_card.save_csv(csv_name)

                # CBOW
                model = CbowModel(corpus_metadata, window_radius, embedding_size)
                csv_name = model.name + '.csv'
                if not ReportCard.saved_with_name(csv_name):
                    model.train()
                    report_card = SynonymTester.administer_tests(model, test_battery)
                    report_card.save_csv(csv_name)


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
