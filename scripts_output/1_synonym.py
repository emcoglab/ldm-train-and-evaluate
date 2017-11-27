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

from .common_output.dataframe import add_model_category_column, add_model_name_column, model_name_without_radius, \
    model_name_without_embedding_size, predict_models_only, model_name_without_distance
from .common_output.figures import cosine_vs_correlation_scores, model_performance_bar_graphs, \
    score_vs_radius_line_graph, score_vs_embedding_size_line_graph, compare_param_values_bf
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
                ticks_as_percentages=True,
                ylim=(0, 1)
            )
            model_performance_bar_graphs(
                results=results_df,
                window_radius=radius,
                key_column_name="Test name",
                test_statistic_name="B10",
                name_prefix="Synonym",
                figures_base_dir=figures_base_dir,
                distance_type=distance_type,
                bayes_factor_graph=True,
                ticks_as_percentages=False
            )

            logger.info(f"Making embedding size line graphs for r={radius} and d={distance_type.name}")
            score_vs_embedding_size_line_graph(
                results=results_df,
                window_radius=radius,
                key_column_name="Test name",
                key_column_values=TEST_NAMES,
                test_statistic_name="Score",
                name_prefix="Synonym",
                figures_base_dir=figures_base_dir,
                distance_type=distance_type,
                ticks_as_percentages=True,
                # Chance line
                additional_h_line_at=0.25,
                ylim=(0, 1)
            )

        logger.info(f"Making score-vs-radius graphs for d={distance_type.name}")
        score_vs_radius_line_graph(
            results=results_df,
            key_column_name="Test name",
            test_statistic_name="Score",
            name_prefix="Synonym",
            figures_base_dir=figures_base_dir,
            distance_type=distance_type,
            ticks_as_percenages=True,
            # Chance line
            additional_h_line_at=0.25,
            ylim=(0, 1)
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

    compare_param_values_bf(
        parameter_name="Window radius",
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
