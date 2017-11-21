"""
===========================
Figures for semantic priming tests.
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
import math
import os
import sys

import numpy
import pandas
import seaborn
from matplotlib import pyplot
from pandas import DataFrame

from .visualisation import add_model_category_column
from ..core.utils.logging import log_message, date_format
from ..core.utils.maths import DistanceType
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)

DV_NAMES = [

    "LDT_200ms_Z",
    "LDT_200ms_Acc",
    # "LDT_1200ms_Z",
    # "LDT_1200ms_Acc",

    "LDT_200ms_Z_Priming",
    "LDT_200ms_Acc_Priming",
    # "LDT_1200ms_Z_Priming",
    # "LDT_1200ms_Acc_Priming",

    "NT_200ms_Z",
    "NT_200ms_Acc",
    # "NT_1200ms_Z",
    # "NT_1200ms_Acc",

    "NT_200ms_Z_Priming",
    "NT_200ms_Acc_Priming",
    # "NT_1200ms_Z_Priming",
    # "NT_1200ms_Acc_Priming"
]

# The Bayes factor threshold at which we say one model is better than another
# This value from Jeffreys (1961) Theory of Probability.
BF_THRESHOLD = math.sqrt(10)


figures_base_dir = os.path.join(Preferences.figures_dir, "priming")


def main():

    regression_results = load_data()

    # Add rsquared increase column
    regression_results["R-squared increase"] = regression_results["Model R-squared"] - regression_results["Baseline R-squared"]

    for radius in Preferences.window_radii:
        for distance_type in DistanceType:
            logger.info(
                f"Making model performance bargraph figures for r={radius}, d={distance_type.name}")
            model_performance_bar_graphs(regression_results, window_radius=radius, distance_type=distance_type)

    for dv_name in DV_NAMES:
        for radius in Preferences.window_radii:
            for corpus_name in ["BNC", "BBC", "UKWAC"]:
                logger.info(f"Making heatmaps dv={dv_name}, r={radius}, c={corpus_name}")
                model_comparison_matrix(regression_results, dv_name, radius, corpus_name)

    # Summary tables
    logger.info("Making top-5 model tables overall")
    table_top_n_models(regression_results, 5)
    for distance_type in DistanceType:
        logger.info(f"Making top-5 model tables overall for {distance_type.name}")
        table_top_n_models(regression_results, 5, distance_type)

    figures_r2_vs_radius(regression_results)


def figures_r2_vs_radius(regression_results: DataFrame):

    figures_dir = os.path.join(figures_base_dir, "effects of radius")

    for distance in [d.name for d in DistanceType]:
        for task_type in ["LDT", "NT"]:
            for y_measure in ["R-squared increase", "B10 approx"]:

                dvs_this_task = [dv for dv in DV_NAMES if dv.startswith(task_type)]

                filtered_df: pandas.DataFrame = regression_results.copy()
                filtered_df = filtered_df[filtered_df["Distance type"] == distance]

                # Filter on task type
                filtered_df = filtered_df[filtered_df["Dependent variable"].isin(dvs_this_task)]

                # Don't need corpus, radius or distance, as they're fixed for each plot
                filtered_df["Model name"] = filtered_df.apply(
                    lambda r:
                    f"{r['Model type']} {r['Embedding size']:.0f}"
                    if r["Model category"] == "Predict"
                    else f"{r['Model type']}",
                    axis=1
                )

                filtered_df = filtered_df.sort_values(by=["Model name", "Window radius"])
                filtered_df = filtered_df.reset_index(drop=True)

                seaborn.set_style("ticks")
                seaborn.set_context(context="paper", font_scale=1)
                grid = seaborn.FacetGrid(
                    data=filtered_df,
                    row="Dependent variable", col="Corpus", hue="Model name",
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
                    size=3.5)
                grid.map(pyplot.plot, "Window radius", y_measure)

                if y_measure is "B10 approx":
                    grid.set(yscale="log")

                # grid.set_titles(row_template="{row_name}", col_template="{col_name}")
                grid.add_legend(bbox_to_anchor=(1, 0.5))

                figure_name = f"priming {distance} {task_type} {y_measure}.png"
                grid.fig.savefig(os.path.join(figures_dir, figure_name), dpi=300)
                pyplot.close(grid.fig)


def table_top_n_models(regression_results_df: DataFrame, top_n: int, distance_type: DistanceType = None):

    summary_dir = Preferences.summary_dir

    results_df = DataFrame()

    for dv_name in DV_NAMES:

        filtered_df: DataFrame = regression_results_df.copy()
        filtered_df = filtered_df[filtered_df["Dependent variable"] == dv_name]

        if distance_type is not None:
            filtered_df = filtered_df[filtered_df["Distance type"] == distance_type.name]

        top_models = filtered_df.sort_values("B10 approx", ascending=False).reset_index(drop=True).head(top_n)

        results_df = results_df.append(top_models)

    if distance_type is None:
        file_name = f"priming_top_{top_n}_models.csv"
    else:
        file_name = f"priming_top_{top_n}_models_{distance_type.name}.csv"

    results_df.to_csv(os.path.join(summary_dir, file_name), index=False)


def model_performance_bar_graphs(spp_results_df: DataFrame, window_radius: int, distance_type: DistanceType):

    figures_dir = os.path.join(figures_base_dir, "model performance bar graphs")

    seaborn.set_style("ticks")

    filtered_df: DataFrame = spp_results_df.copy()
    filtered_df = filtered_df[filtered_df["Window radius"] == window_radius]
    filtered_df = filtered_df[filtered_df["Distance type"] == distance_type.name]

    # Model name doesn't need to include corpus or distance, since those are fixed
    filtered_df["Model name"] = filtered_df.apply(
        lambda r:
        # TODO: embedding sizes aren't ints for some reason, so we have to force this here 🤦‍
        f"{r['Model type']} {r['Embedding size']:.0f}"
        if r['Model category'] == "Predict"
        else f"{r['Model type']}",
        axis=1
    )

    # Get info about the dv
    filtered_df["dv_test_type"] = filtered_df.apply(lambda r: "LDT"     if r["Dependent variable"].startswith("LDT") else "NT",   axis=1)
    filtered_df["dv_measure"]   = filtered_df.apply(lambda r: "Acc"     if "Acc" in r["Dependent variable"]          else "Z-RT", axis=1)
    filtered_df["dv_soa"]       = filtered_df.apply(lambda r: 200       if "_200ms" in r["Dependent variable"]       else 1200,   axis=1)
    filtered_df["dv_priming"]   = filtered_df.apply(lambda r: True      if "Priming" in r["Dependent variable"]      else False,  axis=1)

    for soa in [200, 1200]:
        for test_type in ["LDT", "NT"]:

            # include z-rt/acc and priming/non-priming distinctions in graphs

            dv_name = f"{test_type} {soa}ms"

            double_filtered_df = filtered_df.copy()

            double_filtered_df = double_filtered_df[double_filtered_df["dv_test_type"] == test_type]
            double_filtered_df = double_filtered_df[double_filtered_df["dv_soa"] == soa]

            seaborn.set_context(context="paper", font_scale=1)
            grid = seaborn.FacetGrid(
                double_filtered_df,
                row="Dependent variable", col="Corpus",
                margin_titles=True,
                size=3)

            grid.set_xticklabels(rotation=-90)

            # Plot the bars
            plot = grid.map(seaborn.barplot, "Model name", "B10 approx", order=[
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

            # Plot the 1-line
            grid.map(pyplot.axhline, y=1,              linestyle="solid",  color="xkcd:bright red")
            grid.map(pyplot.axhline, y=BF_THRESHOLD,   linestyle="dotted", color="xkcd:bright red")
            grid.map(pyplot.axhline, y=1/BF_THRESHOLD, linestyle="dotted", color="xkcd:bright red")

            grid.set(yscale="log")

            grid.set_ylabels("BF10")

            pyplot.subplots_adjust(top=0.92)
            grid.fig.suptitle(f"Priming BF10 for {dv_name} radius {window_radius} using {distance_type.name} distance")

            figure_name = f"priming {dv_name} r={window_radius} {distance_type.name}.png"

            # I don't know why PyCharm doesn't find this... it works...
            # noinspection PyUnresolvedReferences
            plot.savefig(os.path.join(figures_dir, figure_name), dpi=300)

            pyplot.close(grid.fig)


def model_comparison_matrix(spp_results_df: DataFrame, dv_name: str, radius: int, corpus_name: str):

    figures_dir = os.path.join(figures_base_dir, "heatmaps all models")

    seaborn.set(style="white")

    filtered_df: DataFrame = spp_results_df.copy()
    filtered_df = filtered_df[filtered_df["Dependent variable"] == dv_name]
    filtered_df = filtered_df[filtered_df["Window radius"] == radius]
    filtered_df = filtered_df[filtered_df["Corpus"] == corpus_name]

    filtered_df["Model name"] = filtered_df.apply(
        lambda r:
        f"{r['Distance type']} {r['Model type']} {r['Embedding size']:.0f}"
        if not numpy.math.isnan(r['Embedding size'])
        else f"{r['Distance type']} {r['Model type']}",
        axis=1
    )

    # filtered_df = filtered_df.sort_values("distance_type")

    # Make the model name the index so it will label the rows and columns of the matrix
    filtered_df = filtered_df.set_index('Model name')

    # filtered_df = filtered_df.sort_values(by=["Model type", "Embedding size", "Distance type"])

    # values - values[:, None] gives col-row
    # which is equivalent to row > col
    bf_matrix = filtered_df["Model BIC"].values - filtered_df["Model BIC"].values[:, None]

    bf_matrix = numpy.exp(bf_matrix)
    bf_matrix = numpy.log10(bf_matrix)
    n_rows, n_columns = bf_matrix.shape

    bf_matrix_df = DataFrame(bf_matrix, filtered_df.index, filtered_df.index)

    # Generate a mask for the upper triangle
    mask = numpy.zeros((n_rows, n_columns), dtype=numpy.bool)
    mask[numpy.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    seaborn.set_context(context="paper", font_scale=1)
    f, ax = pyplot.subplots(figsize=(16, 14))

    # Draw the heatmap with the mask and correct aspect ratio
    seaborn.heatmap(bf_matrix_df,
                    # mask=mask,
                    cmap=seaborn.diverging_palette(250, 15, s=75, l=50, center="dark", as_cmap=True),
                    square=True, cbar_kws={"shrink": .5})

    pyplot.xticks(rotation=-90)
    pyplot.yticks(rotation=0)

    figure_name = f"priming heatmap {dv_name} r={radius} {corpus_name}.png"
    figure_title = f"log(BF(row,col)) for {dv_name} ({corpus_name}, r={radius})"

    ax.set_title(figure_title)

    f.savefig(os.path.join(figures_dir, figure_name), dpi=300)

    pyplot.close(f)


def load_data() -> DataFrame:
    """
    Load a DataFrame from a collection of CSV fragments.
    """
    results_dir = Preferences.spp_results_dir
    separator = ","

    with open(os.path.join(results_dir, "regression.csv"), mode="r", encoding="utf-8") as regression_file:
        regression_df = pandas.read_csv(regression_file, sep=separator, header=0)

    add_model_category_column(regression_df)

    return regression_df


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
