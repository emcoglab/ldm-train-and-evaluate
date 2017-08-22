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
import os
import sys

from ..core.corpus.distribution import FreqDist
from ..core.evaluation.synonym import ToeflTest, EslTest, McqTest, SynonymTester
from ..core.model.count import PPMIModel, LogNgramModel, ConditionalProbabilityModel, ProbabilityRatioModel
from ..core.model.predict import SkipGramModel, CbowModel
from ..core.utils.indexing import TokenIndexDictionary
from ..core.utils.logging import log_message, date_format
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def main():
    report_card_dir = os.path.join(Preferences.eval_dir, "synonyms", "report cards")
    csv_name_pattern = "{model_name}.csv"

    test_battery = [ToeflTest(), EslTest(), McqTest()]

    for corpus_metadata in Preferences.source_corpus_metas:

        token_index = TokenIndexDictionary.load(corpus_metadata.index_path)
        freq_dist = FreqDist.load(corpus_metadata.freq_dist_path)

        tester = SynonymTester(test_battery)

        for window_radius in Preferences.window_radii:

            # COUNT MODELS

            # Log n-gram
            model = LogNgramModel(corpus_metadata, Preferences.model_dir, window_radius, token_index)
            if not tester.all_transcripts_exist_for(model):
                model.train()
                report_card = tester.administer_tests(model)
                report_card.save_csv(os.path.join(report_card_dir, csv_name_pattern.format(model_name=model.name)))
                report_card.save_csv(os.path.join(report_card_dir, csv_name_pattern.format(model_name="0_MASTER")),
                                     append_existing=True)

            # Conditional probability
            model = ConditionalProbabilityModel(corpus_metadata, Preferences.model_dir, window_radius, token_index,
                                                freq_dist)
            if not tester.all_transcripts_exist_for(model):
                model.train()
                report_card = tester.administer_tests(model)
                report_card.save_csv(os.path.join(report_card_dir, csv_name_pattern.format(model_name=model.name)))
                report_card.save_csv(os.path.join(report_card_dir, csv_name_pattern.format(model_name="0_MASTER")),
                                     append_existing=True)

            # Probability ratios
            model = ProbabilityRatioModel(corpus_metadata, Preferences.model_dir, window_radius, token_index,
                                          freq_dist)
            if not tester.all_transcripts_exist_for(model):
                model.train()
                report_card = tester.administer_tests(model)
                report_card.save_csv(os.path.join(report_card_dir, csv_name_pattern.format(model_name=model.name)))
                report_card.save_csv(os.path.join(report_card_dir, csv_name_pattern.format(model_name="0_MASTER")),
                                     append_existing=True)

            # PPMI
            model = PPMIModel(corpus_metadata, Preferences.model_dir, window_radius, token_index, freq_dist)
            if not tester.all_transcripts_exist_for(model):
                model.train()
                report_card = tester.administer_tests(model)
                report_card.save_csv(os.path.join(report_card_dir, csv_name_pattern.format(model_name=model.name)))
                report_card.save_csv(os.path.join(report_card_dir, csv_name_pattern.format(model_name="0_MASTER")),
                                     append_existing=True)

            # PPMI (TRUNCATED, for replication of B&L 2007)
            model = PPMIModel(corpus_metadata, Preferences.model_dir, window_radius, token_index, freq_dist)
            if not tester.all_transcripts_exist_for(model, truncate_vectors_at_length=10_000):
                model.train()
                report_card = tester.administer_tests(model, truncate_vectors_at_length=10_000)
                report_card.save_csv(os.path.join(report_card_dir, csv_name_pattern.format(model_name=model.name
                                                                                           + " (100k)")))
                report_card.save_csv(os.path.join(report_card_dir, csv_name_pattern.format(model_name="0_MASTER")),
                                     append_existing=True)

            # PREDICT MODELS

            for embedding_size in Preferences.predict_embedding_sizes:

                # Skip-gram
                model = SkipGramModel(corpus_metadata, Preferences.model_dir, window_radius, embedding_size)
                if not tester.all_transcripts_exist_for(model):
                    model.train()
                    report_card = tester.administer_tests(model)
                    report_card.save_csv(os.path.join(report_card_dir, csv_name_pattern.format(model_name=model.name)))
                    report_card.save_csv(os.path.join(report_card_dir, csv_name_pattern.format(model_name="0_MASTER")),
                                         append_existing=True)

                # CBOW
                model = CbowModel(corpus_metadata, Preferences.model_dir, window_radius, embedding_size)
                if not tester.all_transcripts_exist_for(model):
                    model.train()
                    report_card = tester.administer_tests(model)
                    report_card.save_csv(os.path.join(report_card_dir, csv_name_pattern.format(model_name=model.name)))
                    report_card.save_csv(os.path.join(report_card_dir, csv_name_pattern.format(model_name="0_MASTER")),
                                         append_existing=True)


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
