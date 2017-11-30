"""
===========================
Figures for similarity judgement tests.
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

from .common_output.dataframe import add_model_category_column, add_model_name_column, predict_models_only
from .common_output.figures import cosine_vs_correlation_scores, model_performance_bar_graphs, \
    score_vs_radius_line_graph, score_vs_embedding_size_line_graph, pearson_vs_spearman_scores, compare_param_values_bf
from .common_output.tables import table_top_n_models
from ..core.evaluation.association import AssociationResults, SimlexSimilarity, WordsimSimilarity, WordsimRelatedness, \
    MenSimilarity, ColourEmotionAssociation, ThematicRelatedness
from ..core.utils.logging import log_message, date_format
from ..core.utils.maths import DistanceType, CorrelationType
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def main():

    def model_name_without_distance(r):
        if r['Model category'] == "Predict":
            return f"{r['Model type']} {r['Embedding size']:.0f} r={r['Window radius']} {r['Corpus']} ({r['Correlation type']})"
        else:
            return f"{r['Model type']} r={r['Window radius']} {r['Corpus']} ({r['Correlation type']})"

    def model_name_without_radius(r):
        if r['Model category'] == "Predict":
            return f"{r['Model type']} {r['Embedding size']:.0f} {r['Distance type']} {r['Corpus']} ({r['Correlation type']})"
        else:
            return f"{r['Model type']} {r['Distance type']} {r['Corpus']} ({r['Correlation type']})"

    def model_name_without_embedding_size(r):
        return f"{r['Model type']} r={r['Window radius']} {r['Distance type']} {r['Corpus']} ({r['Correlation type']})"

    # We make an artificial distinction between similarity data
    # and similarity-based association norms,
    # but we treat them both the same
    for artificial_distinction in ["Similarity", "Norms"]:

        figures_base_dir = os.path.join(Preferences.figures_dir, artificial_distinction.lower())

        if artificial_distinction == "Similarity":
            test_names = [
                SimlexSimilarity().name,
                WordsimSimilarity().name,
                WordsimRelatedness().name,
                MenSimilarity().name
            ]
        elif artificial_distinction == "Norms":
            test_names = [
                # ColourEmotionAssociation().name,
                ThematicRelatedness().name,
                ThematicRelatedness(only_use_response=1).name
            ]
        else:
            raise ValueError()

        results_df = AssociationResults().load().data

        add_model_category_column(results_df)
        add_model_name_column(results_df)

        results_df = results_df[results_df["Test name"].isin(test_names)]

        # Negative correlations are better correlations, so flip the sign for the purposes of display
        results_df["Correlation"] = results_df["Correlation"].apply(lambda r: (-1) * r)

        for correlation_type in CorrelationType:
            for distance_type in DistanceType:

                for radius in Preferences.window_radii:
                    logger.info(f"Making model performance bar graph figures for r={radius}, d={distance_type.name}, c={correlation_type.name}")
                    model_performance_bar_graphs(
                        results=results_df[results_df["Correlation type"] == correlation_type.name],
                        window_radius=radius,
                        key_column_name="Test name",
                        test_statistic_name="Correlation",
                        name_prefix=f"{artificial_distinction} ({correlation_type.name})",
                        figures_base_dir=figures_base_dir,
                        distance_type=distance_type,
                        ylim=(None, 1)
                    )
                    model_performance_bar_graphs(
                        results=results_df[results_df["Correlation type"] == correlation_type.name],
                        window_radius=radius,
                        key_column_name="Test name",
                        test_statistic_name="B10 approx",
                        name_prefix=f"{artificial_distinction} ({correlation_type.name})",
                        figures_base_dir=figures_base_dir,
                        bayes_factor_graph=True,
                        distance_type=distance_type,
                    )

                    logger.info(f"Making embedding size line graphs for r={radius} d={distance_type.name}")
                    score_vs_embedding_size_line_graph(
                        results=results_df[results_df["Correlation type"] == correlation_type.name],
                        window_radius=radius,
                        key_column_name="Test name",
                        key_column_values=test_names,
                        test_statistic_name="Correlation",
                        name_prefix=f"{artificial_distinction} ({correlation_type.name})",
                        figures_base_dir=figures_base_dir,
                        distance_type=distance_type,
                        ylim=(None, 1)
                    )

                logger.info(f"Making correlation-vs-radius figures")
                score_vs_radius_line_graph(
                    results=results_df[results_df["Correlation type"] == correlation_type.name],
                    key_column_name="Test name",
                    test_statistic_name="Correlation",
                    name_prefix=f"{artificial_distinction} ({correlation_type.name})",
                    figures_base_dir=figures_base_dir,
                    distance_type=distance_type,
                    ylim=(None, 1)
                )

            # Summary tables

            logger.info("Making top-5 model tables overall")
            table_top_n_models(
                results=results_df[results_df["Correlation type"] == correlation_type.name],
                top_n=5,
                key_column_values=test_names,
                test_statistic_name="Correlation",
                name_prefix=f"{artificial_distinction} ({correlation_type.name})",
                key_column_name="Test name"
            )
            for distance_type in DistanceType:
                logger.info(f"Making top-5 model tables overall for {distance_type.name}")
                table_top_n_models(
                    results=results_df[results_df["Correlation type"] == correlation_type.name],
                    top_n=5,
                    key_column_values=test_names,
                    test_statistic_name="Correlation",
                    name_prefix=f"{artificial_distinction} ({correlation_type.name})",
                    key_column_name="Test name",
                    distance_type=distance_type
                )

            cosine_vs_correlation_scores(
                results=results_df[results_df["Correlation type"] == correlation_type.name],
                figures_base_dir=figures_base_dir,
                test_names=test_names,
                test_statistic_column_name="Correlation",
                name_prefix=f"{artificial_distinction} ({correlation_type.name})"
            )

            compare_param_values_bf(
                parameter_name="Window radius",
                test_results=results_df[results_df["Correlation type"] == correlation_type.name],
                bf_statistic_name="B10 approx",
                figures_base_dir=figures_base_dir,
                name_prefix=f"{artificial_distinction} ({correlation_type.name})",
                key_column_name="Test name",
                key_column_values=test_names,
                parameter_values=Preferences.window_radii,
                model_name_func=model_name_without_radius
            )
            compare_param_values_bf(
                parameter_name="Embedding size",
                test_results=results_df[results_df["Correlation type"] == correlation_type.name],
                bf_statistic_name="B10 approx",
                figures_base_dir=figures_base_dir,
                name_prefix=f"{artificial_distinction} ({correlation_type.name})",
                key_column_name="Test name",
                key_column_values=test_names,
                parameter_values=Preferences.predict_embedding_sizes,
                model_name_func=model_name_without_embedding_size,
                row_filter=predict_models_only
            )
            compare_param_values_bf(
                parameter_name="Distance type",
                test_results=results_df[results_df["Correlation type"] == correlation_type.name],
                bf_statistic_name="B10 approx",
                figures_base_dir=figures_base_dir,
                name_prefix=f"{artificial_distinction} ({correlation_type.name})",
                key_column_name="Test name",
                key_column_values=test_names,
                parameter_values=[d.name for d in DistanceType],
                model_name_func=model_name_without_distance
            )

        pearson_vs_spearman_scores(
            results=results_df,
            figures_base_dir=figures_base_dir,
            test_names=test_names,
            name_prefix=artificial_distinction
        )


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
