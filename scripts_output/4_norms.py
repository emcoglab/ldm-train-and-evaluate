"""
===========================
Figures for association norms tests.
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

import seaborn
import pandas
from pandas import DataFrame
from matplotlib import pyplot

from ..core.evaluation.association import AssociationResults, ColourEmotionAssociation, ThematicRelatedness
from ..core.utils.logging import log_message, date_format
from ..core.utils.maths import DistanceType, CorrelationType, magnitude_of_negative
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)

TEST_NAMES = [ColourEmotionAssociation().name, ThematicRelatedness().name, ThematicRelatedness(only_use_response=1).name]

figures_base_dir = os.path.join(Preferences.figures_dir, "norms")


def main():
    results_df = AssociationResults().load().data

    results_df["Model category"] = results_df.apply(
        lambda r: "Count" if pandas.isnull(r["Embedding size"]) else "Predict", axis=1)

    results_df["Model name"] = results_df.apply(
        lambda r:
        f"{r['Corpus']} {r['Distance type']} {r['Model type']} r={r['Radius']} {r['Embedding size']:.0f}"
        if r["Model category"] == "Predict"
        else f"{r['Corpus']} {r['Distance type']} {r['Model type']} r={r['Radius']}",
        axis=1
    )

    # We make an artificial distinction between similarity data and similarity-based association norms
    results_df = results_df[results_df["Test name"].isin(TEST_NAMES)]

    logger.info(f"Making correlation-vs-radius figures")
    figures_score_vs_radius(results_df)

    for radius in Preferences.window_radii:
        for distance_type in DistanceType:
            for correlation_type in CorrelationType:
                logger.info(
                    f"Making model performance bar graph figures for r={radius}, d={distance_type.name}, c={correlation_type.name}")
                model_performance_bar_graphs(results_df, window_radius=radius, distance_type=distance_type,
                                             correlation_type=correlation_type)

    # Summary tables
    logger.info("Making top-5 model tables overall")
    table_top_n_models(results_df, 5)
    for distance_type in DistanceType:
        logger.info(f"Making top-5 model tables overall for {distance_type.name}")
        table_top_n_models(results_df, 5, distance_type)

    cos_vs_cor_scores(results_df)


def table_top_n_models(regression_results_df: DataFrame,
                       top_n: int,
                       distance_type: DistanceType = None):
    summary_dir = Preferences.summary_dir

    results_df = DataFrame()

    for correlation_type in CorrelationType:
        for test_name in TEST_NAMES:

            filtered_df: DataFrame = regression_results_df.copy()
            filtered_df = filtered_df[filtered_df["Test name"] == test_name]
            filtered_df = filtered_df[filtered_df["Correlation type"] == correlation_type.name]

            if distance_type is not None:
                filtered_df = filtered_df[filtered_df["Distance type"] == distance_type.name]

            top_models = filtered_df.sort_values("Correlation", ascending=True).reset_index(drop=True).head(top_n)

            results_df = results_df.append(top_models)

    if distance_type is None:
        file_name = f"norms_top_{top_n}_models.csv"
    else:
        file_name = f"norms_top_{top_n}_models_{distance_type.name}.csv"

    results_df.to_csv(os.path.join(summary_dir, file_name), index=False)


def model_performance_bar_graphs(similarity_results: DataFrame, window_radius: int,
                                 distance_type: DistanceType, correlation_type: CorrelationType):
    figures_dir = os.path.join(figures_base_dir, "model performance bar graphs")

    seaborn.set_style("ticks")

    filtered_df: DataFrame = similarity_results.copy()
    filtered_df = filtered_df[filtered_df["Radius"] == window_radius]
    filtered_df = filtered_df[filtered_df["Distance type"] == distance_type.name]
    filtered_df = filtered_df[filtered_df["Correlation type"] == correlation_type.name]

    # Use absolute values of correlation
    similarity_results["Correlation"] = similarity_results.apply(lambda r: magnitude_of_negative(r["Correlation"]))

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
    grid.fig.suptitle(
        f"Model {correlation_type.name} Correlations for Radius {window_radius} using {distance_type.name} distance")

    figure_name = f"norms r={window_radius} {distance_type.name} corr={correlation_type.name}.png"

    # I don't know why PyCharm doesn't find this... it works...
    # noinspection PyUnresolvedReferences
    plot.savefig(os.path.join(figures_dir, figure_name), dpi=300)

    pyplot.close(grid.fig)


def figures_score_vs_radius(similarity_results):
    figures_dir = os.path.join(figures_base_dir, "effects of radius")

    for correlation_type in CorrelationType:

        for distance in [d.name for d in DistanceType]:
            filtered_df: DataFrame = similarity_results.copy()
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

            similarity_results["Correlation"] = similarity_results.apply(
                lambda r: magnitude_of_negative(r["Correlation"]))

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

            figure_name = f"norms {distance} {correlation_type.name}.png"
            grid.fig.savefig(os.path.join(figures_dir, figure_name), dpi=300)
            pyplot.close(grid.fig)


def cos_vs_cor_scores(similarity_results: DataFrame):
    figures_dir = os.path.join(figures_base_dir, "effects of distance")
    seaborn.set(style="white", palette="muted", color_codes=True)

    distribution = []
    for test_name in TEST_NAMES:
        for correlation_type in CorrelationType:

            filtered_df: DataFrame = similarity_results.copy()
            filtered_df = filtered_df[filtered_df["Test name"] == test_name]
            filtered_df = filtered_df[filtered_df["Correlation type"] == correlation_type.name]

            # Use absolute values of correlation
            similarity_results["Correlation"] = similarity_results.apply(
                lambda r: magnitude_of_negative(r["Correlation"]))

            filtered_df["Model name"] = filtered_df.apply(
                lambda r:
                f"{r['Model type']} {r['Embedding size']:.0f} r={r['Radius']} {r['Corpus']}"
                if r['Embedding size'] is not None
                else f"{r['Model type']} r={r['Radius']} {r['Corpus']}",
                axis=1
            )

            for model_name in set(filtered_df["Model name"]):
                cos_df: DataFrame = filtered_df.copy()
                cos_df = cos_df[cos_df["Model name"] == model_name]
                cos_df = cos_df[cos_df["Distance type"] == "cosine"]

                corr_df: DataFrame = filtered_df.copy()
                corr_df = corr_df[corr_df["Model name"] == model_name]
                corr_df = corr_df[corr_df["Distance type"] == "correlation"]

                # barf
                score_cos = list(cos_df["Correlation"])[0]
                score_corr = list(corr_df["Correlation"])[0]

                distribution.append([test_name, correlation_type.name, score_cos, score_corr])

    dist_df = DataFrame(distribution,
                        columns=["Test name", "Correlation type", "Cosine score", "Correlation score"])

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
    grid.fig.suptitle(f"Norms: Correlation- & cosine-distance scores")

    grid.savefig(os.path.join(figures_dir, f"norms scores cos-vs-cor.png"), dpi=300)
    pyplot.close(grid.fig)


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
