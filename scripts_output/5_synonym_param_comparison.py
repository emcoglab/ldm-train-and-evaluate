"""
===========================
Evaluate using synonym test results.
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

from .common_output.figures import compare_param_values_bf
from .common_output.dataframe import model_name_without_radius, model_name_without_embedding_size, \
    model_name_without_distance, predict_models_only, add_model_category_column, add_model_name_column
from ..core.evaluation.synonym import ToeflTest, EslTest, LbmMcqTest, SynonymResults
from ..core.utils.logging import log_message, date_format
from ..core.utils.maths import DistanceType
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)

TEST_NAMES = [ToeflTest().name, EslTest().name, LbmMcqTest().name]

figures_base_dir = os.path.join(Preferences.figures_dir, "synonym")


def main():
    results_df = SynonymResults().load().data

    add_model_category_column(results_df)
    add_model_name_column(results_df)

    compare_param_values_bf(
        parameter_name="Radius",
        test_results=results_df,
        name_prefix="Synonym",
        parameter_values=Preferences.window_radii,
        model_name_func=model_name_without_radius,
        figures_base_dir=figures_base_dir,
        key_column_name="Test name",
        key_column_values=TEST_NAMES,
        bf_statistic_name="B10"
    )
    compare_param_values_bf(
        parameter_name="Embedding size",
        test_results=results_df,
        name_prefix="Synonym",
        parameter_values=Preferences.predict_embedding_sizes,
        model_name_func=model_name_without_embedding_size,
        figures_base_dir=figures_base_dir,
        key_column_name="Test name",
        key_column_values=TEST_NAMES,
        bf_statistic_name="B10",
        row_filter=predict_models_only
    )
    compare_param_values_bf(
        parameter_name="Distance type",
        test_results=results_df,
        name_prefix="Synonym",
        parameter_values=[d.name for d in DistanceType],
        model_name_func=model_name_without_distance,
        figures_base_dir=figures_base_dir,
        key_column_name="Test name",
        key_column_values=TEST_NAMES,
        bf_statistic_name="B10"
    )


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
