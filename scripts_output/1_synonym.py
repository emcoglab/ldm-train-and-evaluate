"""
===========================
Figures for synonym tests.
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

from .common_output.dataframe import add_model_category_column, add_model_name_column
from .common_output.figures import cosine_vs_correlation_scores, model_performance_bar_graphs, \
    score_vs_radius_line_graph, figures_embedding_size
from .common_output.tables import table_top_n_models
from ..core.evaluation.synonym import ToeflTest, LbmMcqTest, EslTest, SynonymResults
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

    for distance_type in DistanceType:
        for radius in Preferences.window_radii:

            logger.info(f"Making model bar graph figures for r={radius} and d={distance_type.name}")
            model_performance_bar_graphs(
                results=results_df,
                window_radius=radius,
                key_column_name="Test name",
                test_statistic_name="Score",
                name_prefix="Synonym",
                figures_base_dir=figures_base_dir,
                distance_type=distance_type,
                extra_h_line_at=0.25,
                ticks_as_percentages=True
            )
            model_performance_bar_graphs(
                results=results_df,
                window_radius=radius,
                key_column_name="Test name",
                test_statistic_name="B10",
                name_prefix="Synonym",
                figures_base_dir=figures_base_dir,
                distance_type=distance_type,
                bayes_factor_decorations=True,
                ticks_as_percentages=False
            )

        for test_name in TEST_NAMES:

            logger.info(f"Making embedding size line graphs for {test_name} d={distance_type.name}")
            figures_embedding_size(
                results=results_df,
                key_column_name="Test name",
                key_column_value=test_name,
                test_statistic_name="Score",
                name_prefix="Synonym",
                figures_base_dir=figures_base_dir,
                distance_type=distance_type,
                ticks_as_percentages=True,
                # Chance line
                additional_h_line_at=0.25
            )

        logger.info(f"Making score-vs-radius graphs for d={distance_type.name}")
        score_vs_radius_line_graph(
            results=results_df,
            key_column_name="Test name",
            test_statistic_name="Score",
            name_prefix="Synonym",
            figures_base_dir=figures_base_dir,
            distance_type=distance_type,
            ticks_as_percenages=True
        )

    # Summary tables
    logger.info("Making top-5 model tables overall")
    table_top_n_models(
        results=results_df,
        top_n=5,
        key_column_values=TEST_NAMES,
        test_statistic_name="Score",
        name_prefix="Synonym",
        key_column_name="Test name"
    )
    for distance_type in DistanceType:
        logger.info(f"Making top-5 model tables overall for {distance_type.name}")
        table_top_n_models(
            results=results_df,
            top_n=5,
            key_column_values=TEST_NAMES,
            test_statistic_name="Score",
            name_prefix="Synonym",
            key_column_name="Test name",
            distance_type=distance_type
        )

    cosine_vs_correlation_scores(results_df, figures_base_dir, TEST_NAMES, "Score", "Synonym", ticks_as_percentages=True)


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
