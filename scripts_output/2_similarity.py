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


def main():

    results_df = AssociationResults().load().data

    results_df["Model category"] = results_df.apply(lambda r: "Count" if pandas.isnull(r["Embedding size"]) else "Predict", axis=1)

    results_df["Model name"] = results_df.apply(
        lambda r:
        f"{r['Corpus']} {r['Distance type']} {r['Model type']} r={r['Radius']} {r['Embedding size']:.0f}"
        if r["Model category"] == "Predict"
        else f"{r['Corpus']} {r['Distance type']} {r['Model type']} r={r['Radius']}",
        axis=1
    )

    logger.info(f"Making correlation-vs-radius figures")
    figures_score_vs_radius(results_df)

    for radius in Preferences.window_radii:
        for distance_type in DistanceType:
            for correlation_type in CorrelationType:
                logger.info(f"Making model performance bar graph figures for r={radius}, d={distance_type.name}, c={correlation_type.name}")
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
            filtered_df = filtered_df[filtered_df["Test name"] == test_name]
            filtered_df = filtered_df[filtered_df["Correlation type"] == correlation_type.name]

            if distance_type is not None:
                filtered_df = filtered_df[filtered_df["Distance type"] == distance_type.name]

            top_models = filtered_df.sort_values("Correlation", ascending=True).reset_index(drop=True).head(top_n)

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
    filtered_df = filtered_df[filtered_df["Radius"] == window_radius]
    filtered_df = filtered_df[filtered_df["Distance type"] == distance_type.name]
    filtered_df = filtered_df[filtered_df["Correlation type"] == correlation_type.name]

    # Use absolute values of correlation
    filtered_df["Correlation"] = abs(filtered_df["Correlation"])

    # Model name doesn't need to include corpus or distance, since those are fixed
    filtered_df["Model name"] = filtered_df.apply(
        lambda r:
        f"{r['Model type']} {r['Embedding size']}"
        if not numpy.math.isnan(r['Embedding size'])
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

    # TODO: this isn't working for some reason
    # Remove the "corpus = " from the titles
    # grid.set_titles(col_template='{col_name}', row_template="{row_name}")

    grid.set_ylabels("Correlation")

    pyplot.subplots_adjust(top=0.92)
    grid.fig.suptitle(f"Model {correlation_type.name} Correlations for Radius {window_radius} using {distance_type.name} distance")

    figure_name = f"similarity r={window_radius} {distance_type.name} corr={correlation_type.name}.png"

    # I don't know why PyCharm doesn't find this... it works...
    # noinspection PyUnresolvedReferences
    plot.savefig(os.path.join(figures_dir, figure_name), dpi=300)

    pyplot.close(grid.fig)


def figures_score_vs_radius(similarity_results):

    figures_dir = os.path.join(figures_base_dir, "effects of Radius")

    for distance in [d.name for d in DistanceType]:

        filtered_df: pandas.DataFrame = similarity_results.copy()
        filtered_df = filtered_df[filtered_df["Distance type"] == distance]
        filtered_df = filtered_df[filtered_df["Correlation type"] == "Spearman"]

        # Don't need corpus, radius or distance, as they're fixed for each plot
        filtered_df["Model name"] = filtered_df.apply(
            lambda r:
            f"{r['Model type']} {r['Embedding size']:.0f}"
            if r["Model category"] == "Predict"
            else f"{r['Model type']}",
            axis=1
        )

        filtered_df["Correlation"] = filtered_df.apply(lambda r: math.fabs(r["Correlation"]), axis=1)

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

        figure_name = f"similarity {distance}.png"
        grid.fig.savefig(os.path.join(figures_dir, figure_name), dpi=300)
        pyplot.close(grid.fig)


def cos_vs_cor_scores(results_df: pandas.DataFrame):

    figures_dir = os.path.join(figures_base_dir, "effects of distance")
    seaborn.set(style="white", palette="muted", color_codes=True)

    distribution = []
    for test_name in TEST_NAMES:
        for correlation_type in CorrelationType:

            filtered_df: pandas.DataFrame = results_df.copy()
            filtered_df = filtered_df[filtered_df["Test name"] == test_name]
            filtered_df = filtered_df[filtered_df["Correlation type"] == correlation_type.name]

            filtered_df["Model name"] = filtered_df.apply(
                lambda r:
                f"{r['Model type']} {r['Embedding size']:.0f} r={r['Radius']} {r['Corpus']}"
                if r['Embedding size'] is not None
                else f"{r['Model type']} r={r['Radius']} {r['Corpus']}",
                axis=1
            )

            for model_name in set(filtered_df["Model name"]):
                cos_df: pandas.DataFrame = filtered_df.copy()
                cos_df = cos_df[cos_df["Model name"] == model_name]
                cos_df = cos_df[cos_df["Distance type"] == "cosine"]

                corr_df: pandas.DataFrame = filtered_df.copy()
                corr_df = corr_df[corr_df["Model name"] == model_name]
                corr_df = corr_df[corr_df["Distance type"] == "Correlation"]

                # barf
                score_cos = math.fabs(list(cos_df["Correlation"])[0])
                score_corr = math.fabs(list(corr_df["Correlation"])[0])

                distribution.append([test_name, correlation_type.name, score_cos, score_corr])

    dist_df = pandas.DataFrame(distribution, columns=["Test name", "Correlation type", "Cosine score", "Correlation score"])

    seaborn.set_context(context="paper", font_scale=1)

    grid = seaborn.FacetGrid(data=dist_df,
                             row="Test name", col="Correlation type",
                             size=5, aspect=1,
                             margin_titles=True,
                             xlim=(0, 1), ylim=(0, 1))

    grid.map(pyplot.scatter, "Cosine score", "Correlation score")

    for ax in grid.axes.flat:
        ax.plot((0, 1), (0, 1), c="r", ls="-")

    pyplot.subplots_adjust(top=0.92)
    grid.fig.suptitle(f"Similarity judgements: Correlation- & cosine-distance scores")

    grid.savefig(os.path.join(figures_dir, f"similarity scores cos-vs-cor.png"), dpi=300)
    pyplot.close(grid.fig)


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
