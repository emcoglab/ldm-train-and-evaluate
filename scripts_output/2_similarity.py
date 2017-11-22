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

import pandas
import seaborn
from matplotlib import pyplot

from ..preferences.preferences import Preferences
from ..core.evaluation.association import AssociationResults, SimlexSimilarity, WordsimSimilarity, WordsimRelatedness, \
    MenSimilarity
from ..core.utils.logging import log_message, date_format
from ..core.utils.maths import DistanceType, CorrelationType, magnitude_of_negative
from ..core.output.dataframe import add_model_category_column, add_model_name_column
from ..core.output.figures import cosine_vs_correlation_scores
from ..core.output.tables import table_top_n_models

logger = logging.getLogger(__name__)

TEST_NAMES = [SimlexSimilarity().name, WordsimSimilarity().name, WordsimRelatedness().name, MenSimilarity().name]

figures_base_dir = os.path.join(Preferences.figures_dir, "similarity")


def main():

    results_df = AssociationResults().load().data

    add_model_category_column(results_df)
    add_model_name_column(results_df)

    # We make an artificial distinction between similarity data and similarity-based association norms
    results_df = results_df[results_df["Test name"].isin(TEST_NAMES)]

    logger.info(f"Making correlation-vs-radius figures")
    figures_score_vs_radius(results_df)

    for radius in Preferences.window_radii:
        for distance_type in DistanceType:
            for correlation_type in CorrelationType:
                logger.info(f"Making model performance bar graph figures for r={radius}, d={distance_type.name}, c={correlation_type.name}")
                model_performance_bar_graphs(results_df, window_radius=radius, distance_type=distance_type, correlation_type=correlation_type)

    # Summary tables
    logger.info("Making top-5 model tables overall")
    for correlation_type in CorrelationType:
        table_top_n_models(
            results=results_df[results_df["Correlation type"] == correlation_type.name],
            top_n=5,
            key_column_values=TEST_NAMES,
            test_statistic_name="Correlation",
            name_prefix=f"Similarity judgements ({correlation_type.name})",
            key_column_name="Test name"
        )
        for distance_type in DistanceType:
            logger.info(f"Making top-5 model tables overall for {distance_type.name}")
            table_top_n_models(
                results=results_df[results_df["Correlation type"] == correlation_type.name],
                top_n=5,
                key_column_values=TEST_NAMES,
                test_statistic_name="Correlation",
                name_prefix=f"Similarity judgements ({correlation_type.name})",
                key_column_name="Test name",
                distance_type=distance_type
            )

    for correlation_type in CorrelationType:
        cos_cor_df = results_df.copy()
        cos_cor_df["Correlation"] = cos_cor_df["Correlation"].apply(magnitude_of_negative, axis=1)
        cosine_vs_correlation_scores(
            results=cos_cor_df[cos_cor_df["Correlation type"] == correlation_type.name],
            figures_base_dir=figures_base_dir,
            test_names=TEST_NAMES,
            test_statistic_column_name="Correlation",
            name_prefix=f"Similarity judgements ({correlation_type.name})"
        )


def model_performance_bar_graphs(similarity_results: pandas.DataFrame, window_radius: int, distance_type: DistanceType, correlation_type: CorrelationType):

    figures_dir = os.path.join(figures_base_dir, "model performance bar graphs")

    seaborn.set_style("ticks")

    filtered_df: pandas.DataFrame = similarity_results.copy()
    filtered_df = filtered_df[filtered_df["Radius"] == window_radius]
    filtered_df = filtered_df[filtered_df["Distance type"] == distance_type.name]
    filtered_df = filtered_df[filtered_df["Correlation type"] == correlation_type.name]

    # Use absolute values of correlation
    filtered_df["Correlation"] = filtered_df["Correlation"].apply(magnitude_of_negative)

    # Model name doesn't need to include corpus or distance, since those are fixed for each sub-plot
    filtered_df["Model name"] = filtered_df.apply(
        lambda r:
        f"{r['Model type']} {r['Embedding size']:.0f}"
        if r['Model category'] == "Predict"
        else f"{r['Model type']}",
        axis=1
    )

    seaborn.set_context(context="paper", font_scale=1)
    grid = seaborn.FacetGrid(
        filtered_df,
        row="Test name", col="Corpus",
        margin_titles=True,
        size=2.5,
        ylim=(0, 1))

    grid.set_xticklabels(rotation=-90)

    # Plot the bars
    plot = grid.map(seaborn.barplot, "Model name", "Correlation", order=[
        "log n-gram",
        "Conditional probability",
        "Probability ratio",
        "PPMI",
        "Skip-gram 50",
        "Skip-gram 100",
        "Skip-gram 200",
        "Skip-gram 300",
        "Skip-gram 500",
        "CBOW 50",
        "CBOW 100",
        "CBOW 200",
        "CBOW 300",
        "CBOW 500",
    ])

    # Plot the chance line
    grid.map(pyplot.axhline, y=0.0, linestyle="solid", color="xkcd:bright red")

    grid.set_ylabels("Correlation")

    pyplot.subplots_adjust(top=0.92)
    grid.fig.suptitle(f"Model {correlation_type.name} Correlations for Radius {window_radius} using {distance_type.name} distance")

    figure_name = f"similarity r={window_radius} {distance_type.name} corr={correlation_type.name}.png"

    # I don't know why PyCharm doesn't find this... it works...
    # noinspection PyUnresolvedReferences
    plot.savefig(os.path.join(figures_dir, figure_name), dpi=300)

    pyplot.close(grid.fig)


def figures_score_vs_radius(similarity_results):

    figures_dir = os.path.join(figures_base_dir, "effects of radius")

    for correlation_type in CorrelationType:
        for distance in [d.name for d in DistanceType]:

            filtered_df: pandas.DataFrame = similarity_results.copy()
            filtered_df = filtered_df[filtered_df["Distance type"] == distance]
            filtered_df = filtered_df[filtered_df["Correlation type"] == correlation_type.name]

            # Don't need corpus, radius or distance, as they're fixed for each plot
            filtered_df["Model name"] = filtered_df.apply(
                lambda r:
                f"{r['Model type']} {r['Embedding size']:.0f}"
                if r["Model category"] == "Predict"
                else f"{r['Model type']}",
                axis=1
            )

            filtered_df["Correlation"] = filtered_df.apply(lambda r: magnitude_of_negative(r["Correlation"]), axis=1)

            filtered_df = filtered_df.sort_values(by=["Model name", "Radius"])
            filtered_df = filtered_df.reset_index(drop=True)

            seaborn.set_style("ticks")
            seaborn.set_context(context="paper", font_scale=1)
            grid = seaborn.FacetGrid(
                data=filtered_df,
                row="Test name", col="Corpus", hue="Model name",
                hue_order=[
                    "log n-gram",
                    "Conditional probability",
                    "Probability ratio",
                    "PPMI",
                    "Skip-gram 50",
                    "Skip-gram 100",
                    "Skip-gram 200",
                    "Skip-gram 300",
                    "Skip-gram 500",
                    "CBOW 50",
                    "CBOW 100",
                    "CBOW 200",
                    "CBOW 300",
                    "CBOW 500"
                ],
                palette=[
                    "orange",
                    "turquoise",
                    "pink",
                    "red",
                    "#0000ff",
                    "#2a2aff",
                    "#5454ff",
                    "#7e7eff",
                    "#a8a8ff",
                    "#00ff00",
                    "#2aff2a",
                    "#54ff54",
                    "#7eff7e",
                    "#a8ffa8",
                ],
                hue_kws=dict(
                    marker=[
                        "o",
                        "o",
                        "o",
                        "o",
                        "^",
                        "^",
                        "^",
                        "^",
                        "^",
                        "^",
                        "^",
                        "^",
                        "^",
                        "^",
                    ]
                ),
                margin_titles=True,
                legend_out=True,
                size=3.5,
                ylim=(0, 1))
            grid.map(pyplot.plot, "Radius", "Correlation")

            grid.add_legend(bbox_to_anchor=(1, 0.5))

            figure_name = f"similarity {distance} {correlation_type.name}.png"
            grid.fig.savefig(os.path.join(figures_dir, figure_name), dpi=300)
            pyplot.close(grid.fig)


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
