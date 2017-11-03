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
import numpy

from matplotlib import pyplot

from ..core.utils.logging import log_message, date_format
from ..core.utils.maths import CorrelationType, DistanceType
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)

DV_NAMES = [
    # LDT
    "LDT_200ms_Z",
    "LDT_200ms_Acc",
    "LDT_1200ms_Z",
    "LDT_1200ms_Acc",
    # NT
    "NT_200ms_Z",
    "NT_200ms_Acc",
    "NT_1200ms_Z",
    "NT_1200ms_Acc",
    # LDT priming
    "LDT_200ms_Z_Priming",
    "LDT_200ms_Acc_Priming",
    "LDT_1200ms_Z_Priming",
    "LDT_1200ms_Acc_Priming",
    # NT priming
    "NT_200ms_Z_Priming",
    "NT_200ms_Acc_Priming",
    "NT_1200ms_Z_Priming",
    "NT_1200ms_Acc_Priming"
]


def ensure_column_safety(df: pandas.DataFrame) -> pandas.DataFrame:
    return df.rename(columns=lambda col_name: col_name.replace(" ", "_").lower())


# TODO: essentially duplicated code
def main():

    spp_results_df = load_data()
    spp_results_df = ensure_column_safety(spp_results_df)

    # Add rsquared increase column
    spp_results_df["r-squared_increase"] = spp_results_df["model_r-squared"] - spp_results_df["baseline_r-squared"]

    for radius in Preferences.window_radii:
        for distance_type in DistanceType:
            logger.info(
                f"Making model performance bargraph figures for r={radius}, d={distance_type.name}")
            model_performance_bar_graphs(spp_results_df, window_radius=radius, distance_type=distance_type)

    for dv_name in DV_NAMES:
        for radius in Preferences.window_radii:
            for corpus_name in ["BNC", "BBC", "UKWAC"]:
                logger.info(f"Making heatmaps dv={dv_name}, r={radius}, c={corpus_name}")
                model_comparison_matrix(spp_results_df, dv_name, radius, corpus_name)

    logger.info(f"Making best-model table")
    best_model_table(spp_results_df)


def model_performance_bar_graphs(spp_results_df: pandas.DataFrame, window_radius: int, distance_type: DistanceType):

    figures_dir = Preferences.figures_dir
    seaborn.set_style("ticks")

    filtered_df: pandas.DataFrame = spp_results_df.copy()
    filtered_df = filtered_df[filtered_df["window_radius"] == window_radius]
    filtered_df = filtered_df[filtered_df["distance_type"] == distance_type.name]

    # Model name doesn't need to include corpus or distance, since those are fixed
    filtered_df["model_name"] = filtered_df.apply(
        lambda r:
        # TODO: embedding sizes aren't ints for some reason, so we have to force this here 🤦‍
        f"{r['model_type']} {r['embedding_size']:.0f}"
        if not numpy.math.isnan(r['embedding_size'])
        else f"{r['model_type']}",
        axis=1
    )

    # Get info about the dv
    filtered_df["dv_test_type"] = filtered_df.apply(lambda r: "LDT"     if r["dependent_variable"].startswith("LDT") else "NT",   axis=1)
    filtered_df["dv_measure"]   = filtered_df.apply(lambda r: "Acc"     if "Acc" in r["dependent_variable"]          else "Z-RT", axis=1)
    filtered_df["dv_soa"]       = filtered_df.apply(lambda r: 200       if "_200ms" in r["dependent_variable"]       else 1200,   axis=1)
    filtered_df["dv_priming"]   = filtered_df.apply(lambda r: "priming" if "Priming" in r["dependent_variable"]      else "",     axis=1)

    for soa in [200, 1200]:
        for test_type in ["LDT", "NT"]:

            dv_name = f"{test_type} {soa}ms"

            double_filtered_df = filtered_df.copy()

            double_filtered_df = double_filtered_df[double_filtered_df["dv_test_type"] == test_type]
            double_filtered_df = double_filtered_df[double_filtered_df["dv_soa"] == soa]

            seaborn.set_context(context="paper", font_scale=1)
            grid = seaborn.FacetGrid(
                double_filtered_df,
                row="dependent_variable", col="corpus",
                margin_titles=True,
                size=3)

            grid.set_xticklabels(rotation=-90)

            # Plot the bars
            plot = grid.map(seaborn.barplot, "model_name", "b10_approx", order=[
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
            grid.map(pyplot.axhline, y=1, linestyle="solid", color="xkcd:bright red")

            grid.set(yscale="log")

            # TODO: this isn't working for some reason
            # Remove the "corpus = " from the titles
            # grid.set_titles(col_template='{col_name}')

            grid.set_ylabels("BF10")

            pyplot.subplots_adjust(top=0.92)
            grid.fig.suptitle(f"Priming BF10 for {dv_name} radius {window_radius} using {distance_type.name} distance")

            figure_name = f"priming {dv_name} r={window_radius} {distance_type.name}.png"

            # I don't know why PyCharm doesn't find this... it works...
            # noinspection PyUnresolvedReferences
            plot.savefig(os.path.join(figures_dir, figure_name), dpi=300)


def model_comparison_matrix(spp_results_df: pandas.DataFrame, dv_name: str, radius: int, corpus_name: str):

    figures_dir = Preferences.figures_dir

    seaborn.set(style="white")

    filtered_df: pandas.DataFrame = spp_results_df.copy()
    filtered_df = filtered_df[filtered_df["dependent_variable"] == dv_name]
    filtered_df = filtered_df[filtered_df["window_radius"] == radius]
    filtered_df = filtered_df[filtered_df["corpus"] == corpus_name]

    filtered_df["model_name"] = filtered_df.apply(
        lambda r:
        f"{r['distance_type']} {r['model_type']} {r['embedding_size']:.0f}"
        if not numpy.math.isnan(r['embedding_size'])
        else f"{r['distance_type']} {r['model_type']}",
        axis=1
    )

    # filtered_df = filtered_df.sort_values("distance_type")

    # Make the model name the index so it will label the rows and columns of the matrix
    filtered_df = filtered_df.set_index('model_name')

    # filtered_df = filtered_df.sort_values(by=["model_type", "embedding_size", "distance_type"])

    # values - values[:, None] gives col-row
    # which is equivalent to row > col
    bf_matrix = filtered_df.model_bic.values - filtered_df.model_bic.values[:, None]

    bf_matrix = numpy.exp(bf_matrix)
    bf_matrix = numpy.log10(bf_matrix)
    n_rows, n_columns = bf_matrix.shape

    bf_matrix_df = pandas.DataFrame(bf_matrix, filtered_df.index, filtered_df.index)

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

    figure_name = f"priming heatmap {dv_name} r={radius} {corpus_name}.png"
    figure_title = f"log(BF(row,col)) for {dv_name} ({corpus_name}, r={radius})"

    ax.set_title(figure_title)

    f.savefig(os.path.join(figures_dir, figure_name), dpi=300)

    pyplot.close(f)


def best_model_table(spp_results: pandas.DataFrame):
    summary_dir = Preferences.summary_dir

    results_df = pandas.DataFrame()

    for dv_name in DV_NAMES:

        filtered_df: pandas.DataFrame = spp_results.copy()
        filtered_df = filtered_df[filtered_df["dependent_variable"] == dv_name]

        best_bf = filtered_df["b10_approx"].max()

        best_models_df = filtered_df[filtered_df["b10_approx"] == best_bf]

        results_df = results_df.append(best_models_df)

    results_df = results_df.reset_index(drop=True)

    results_df.to_csv(os.path.join(summary_dir, f"priming_best_models.csv"))


def load_data() -> pandas.DataFrame:
    """
    Load a pandas.DataFrame from a collection of CSV fragments.
    """
    results_dir = Preferences.spp_results_dir
    separator = ","

    with open(os.path.join(results_dir, "regression.csv"), mode="r", encoding="utf-8") as regression_file:
        regression_df = pandas.read_csv(regression_file, sep=separator, header=0)

    return regression_df


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
