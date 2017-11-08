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
import math

import numpy
import pandas
import seaborn

from matplotlib import pyplot

from ..core.evaluation.association import SimlexSimilarity, WordsimSimilarity, WordsimRelatedness, MenSimilarity, \
    AssociationResults
from ..core.utils.logging import log_message, date_format
from ..core.utils.maths import DistanceType, CorrelationType
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)

TEST_NAMES = [SimlexSimilarity().name, WordsimSimilarity().name, WordsimRelatedness().name, MenSimilarity().name]

figures_base_dir = os.path.join(Preferences.figures_dir, "similarity")


def ensure_column_safety(df: pandas.DataFrame) -> pandas.DataFrame:
    return df.rename(columns=lambda col_name: col_name.replace(" ", "_").lower())


def main():

    results_df = AssociationResults().load().data
    results_df = ensure_column_safety(results_df)

    results_df["model"] = results_df.apply(
        lambda r:
        f"{r['corpus']} {r['distance_type']} {r['model_type']} {r['embedding_size']}"
        if r['embedding_size'] is not None
        else f"{r['corpus']} {r['distance_type']} {r['model_type']}",
        axis=1
    )

    for test_name in TEST_NAMES:
        logger.info(f"Making score-vs-radius figures for {test_name}")
        figures_score_vs_radius(results_df, test_name)

    for radius in Preferences.window_radii:
        for distance_type in DistanceType:
            for correlation_type in CorrelationType:
                logger.info(f"Making model performance bargraph figures for r={radius}, d={distance_type.name}, c={correlation_type.name}")
                model_performance_bar_graphs(results_df, window_radius=radius, distance_type=distance_type, correlation_type=correlation_type)

    # Summary tables
    logger.info("Making top-5 model tables overall")
    table_top_n_models(results_df, 5)
    for distance_type in DistanceType:
        logger.info(f"Making top-5 model tables overall for {distance_type.name}")
        table_top_n_models(results_df, 5, distance_type)

    cos_vs_cor_scores(results_df)


def table_top_n_models(regression_results_df: pandas.DataFrame,
                       top_n: int,
                       distance_type: DistanceType = None):

    summary_dir = Preferences.summary_dir

    results_df = pandas.DataFrame()

    for correlation_type in CorrelationType:
        for test_name in TEST_NAMES:

            filtered_df: pandas.DataFrame = regression_results_df.copy()
            filtered_df = filtered_df[filtered_df["test_name"] == test_name]
            filtered_df = filtered_df[filtered_df["correlation_type"] == correlation_type.name]

            if distance_type is not None:
                filtered_df = filtered_df[filtered_df["distance_type"] == distance_type.name]

            top_models = filtered_df.sort_values("correlation", ascending=True).reset_index(drop=True).head(top_n)

            results_df = results_df.append(top_models)

    if distance_type is None:
        file_name = f"similarity_top_{top_n}_models.csv"
    else:
        file_name = f"similarity_top_{top_n}_models_{distance_type.name}.csv"

    results_df.to_csv(os.path.join(summary_dir, file_name), index=False)


def model_performance_bar_graphs(similarity_results_df: pandas.DataFrame, window_radius: int, distance_type: DistanceType, correlation_type: CorrelationType):

    figures_dir = os.path.join(figures_base_dir, "model performance bar graphs")

    seaborn.set_style("ticks")

    filtered_df: pandas.DataFrame = similarity_results_df.copy()
    filtered_df = filtered_df[filtered_df["radius"] == window_radius]
    filtered_df = filtered_df[filtered_df["distance_type"] == distance_type.name]
    filtered_df = filtered_df[filtered_df["correlation_type"] == correlation_type.name]

    # Use absolute values of correlation
    filtered_df["correlation"] = abs(filtered_df["correlation"])

    # Model name doesn't need to include corpus or distance, since those are fixed
    filtered_df["model_name"] = filtered_df.apply(
        lambda r:
        f"{r['model_type']} {r['embedding_size']}"
        if not numpy.math.isnan(r['embedding_size'])
        else f"{r['model_type']}",
        axis=1
    )

    seaborn.set_context(context="paper", font_scale=1)
    grid = seaborn.FacetGrid(
        filtered_df,
        row="test_name", col="corpus",
        margin_titles=True,
        size=2.5,
        ylim=(0, 1))

    grid.set_xticklabels(rotation=-90)

    # Plot the bars
    plot = grid.map(seaborn.barplot, "model_name", "correlation", order=[
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

    # TODO: this isn't working for some reason
    # Remove the "corpus = " from the titles
    # grid.set_titles(col_template='{col_name}', row_template="{row_name}")

    grid.set_ylabels("Correlation")

    pyplot.subplots_adjust(top=0.92)
    grid.fig.suptitle(f"Model {correlation_type.name} correlations for radius {window_radius} using {distance_type.name} distance")

    figure_name = f"similarity r={window_radius} {distance_type.name} corr={correlation_type.name}.png"

    # I don't know why PyCharm doesn't find this... it works...
    # noinspection PyUnresolvedReferences
    plot.savefig(os.path.join(figures_dir, figure_name), dpi=300)

    pyplot.close(grid.fig)


def figures_score_vs_radius(similarity_results, test_name):

    figures_dir = os.path.join(figures_base_dir, "effects of radius")

    for distance in [d.name for d in DistanceType]:
        for corpus in ["BNC", "BBC", "UKWAC"]:
            figure_name = f"similarity {test_name} {corpus} {distance}.png"

            filtered_dataframe: pandas.DataFrame = similarity_results.copy()
            filtered_dataframe = filtered_dataframe[filtered_dataframe["corpus"] == corpus]
            filtered_dataframe = filtered_dataframe[filtered_dataframe["distance_type"] == distance]
            filtered_dataframe = filtered_dataframe[filtered_dataframe["test_name"] == test_name]

            filtered_dataframe = filtered_dataframe.sort_values(by=["model", "radius"])
            filtered_dataframe = filtered_dataframe.reset_index(drop=True)

            filtered_dataframe = filtered_dataframe[[
                "model",
                "radius",
                "correlation"
            ]]

            plot = seaborn.factorplot(data=filtered_dataframe,
                                      x="radius", y="correlation",
                                      hue="model",
                                      size=7, aspect=1.8,
                                      legend=False)

            plot.set(ylim=(-1, 1))

            # Put the legend out of the figure
            # resize figure box to -> put the legend out of the figure
            plot_box = plot.ax.get_position()  # get position of figure
            plot.ax.set_position([plot_box.x0, plot_box.y0, plot_box.width * 0.75, plot_box.height])  # resize position

            # Put a legend to the right side
            plot.ax.legend(loc='center right', bbox_to_anchor=(1.35, 0.5), ncol=1)

            plot.savefig(os.path.join(figures_dir, figure_name))

            pyplot.close(plot.fig)


def cos_vs_cor_scores(results_df: pandas.DataFrame):

    figures_dir = os.path.join(figures_base_dir, "effects of distance")
    seaborn.set(style="white", palette="muted", color_codes=True)

    distribution = []
    for test_name in TEST_NAMES:
        for correlation_type in CorrelationType:

            filtered_df: pandas.DataFrame = results_df.copy()
            filtered_df = filtered_df[filtered_df["test_name"] == test_name]
            filtered_df = filtered_df[filtered_df["correlation_type"] == correlation_type.name]

            filtered_df["model_name"] = filtered_df.apply(
                lambda r:
                f"{r['model_type']} {r['embedding_size']:.0f} r={r['radius']} {r['corpus']}"
                if r['embedding_size'] is not None
                else f"{r['model_type']} r={r['radius']} {r['corpus']}",
                axis=1
            )

            for model_name in set(filtered_df["model_name"]):
                cos_df: pandas.DataFrame = filtered_df.copy()
                cos_df = cos_df[cos_df["model_name"] == model_name]
                cos_df = cos_df[cos_df["distance_type"] == "cosine"]

                corr_df: pandas.DataFrame = filtered_df.copy()
                corr_df = corr_df[corr_df["model_name"] == model_name]
                corr_df = corr_df[corr_df["distance_type"] == "correlation"]

                # barf
                score_cos = math.fabs(list(cos_df["correlation"])[0])
                score_corr = math.fabs(list(corr_df["correlation"])[0])

                distribution.append([test_name, correlation_type.name, score_cos, score_corr])

    dist_df = pandas.DataFrame(distribution, columns=["Test name", "Correlation type", "Cosine score", "Correlation score"])

    seaborn.set_context(context="paper", font_scale=1)

    grid = seaborn.FacetGrid(data=dist_df,
                             row="Test name", col="Correlation type",
                             size=2, aspect=1,
                             margin_titles=True,
                             xlim=(-1, 1), ylim=(-1, 1))

    grid.map(pyplot.scatter, "Cosine score", "Correlation score")

    for ax in grid.axes.flat:
        ax.plot((-1, 1), (-1, 1), c="r", ls="-")

    pyplot.subplots_adjust(top=0.92)
    grid.fig.suptitle(f"Similarity judgements: correlation- & cosine-distance scores")

    grid.savefig(os.path.join(figures_dir, f"similarity scores cos-vs-cor.png"), dpi=300)
    pyplot.close(grid.fig)


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
