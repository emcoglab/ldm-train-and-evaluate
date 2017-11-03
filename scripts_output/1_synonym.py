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

from glob import glob

import numpy
import pandas
import seaborn

from matplotlib import pyplot

from ..core.utils.logging import log_message, date_format
from ..core.utils.maths import DistanceType
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)

TEST_NAMES = ["TOEFL", "ESL", "LBM's new MCQ"]


def main():
    synonym_results = load_data()
    synonym_results = ensure_column_safety(synonym_results)

    synonym_results["model_name"] = synonym_results.apply(
        lambda r:
        f"{r['corpus']} {r['distance']} {r['model']} {r['embedding_size']}"
        if not numpy.math.isnan(r['embedding_size'])
        else f"{r['corpus']} {r['distance']} {r['model']}",
        axis=1
    )

    for test_name in TEST_NAMES:
        logger.info(f"Making score-vs-radius figures for {test_name}")
        figures_score_vs_radius(synonym_results, test_name)
        logger.info(f"Making embedding size figures for {test_name}")
        figures_embedding_size(synonym_results, test_name)

    for radius in Preferences.window_radii:
        for distance_type in DistanceType:
            logger.info(f"Making model bar graph figures for r={radius} and d={distance_type.name}")
            model_performance_bar_graphs(synonym_results, window_radius=radius, distance_type=distance_type)

    summary_tables(synonym_results)


def ensure_column_safety(df: pandas.DataFrame) -> pandas.DataFrame:
    return df.rename(columns=lambda col_name: col_name.replace(" ", "_").lower())


def summary_tables(regression_results_df: pandas.DataFrame):
    summary_dir = Preferences.summary_dir

    results_df = pandas.DataFrame()

    for test_name in TEST_NAMES:

        filtered_df: pandas.DataFrame = regression_results_df.copy()
        filtered_df = filtered_df[filtered_df["test_name"] == test_name]

        best_score = filtered_df["score"].max()

        best_models_df = filtered_df[filtered_df["score"] == best_score]

        results_df = results_df.append(best_models_df)

    results_df = results_df.reset_index(drop=True)

    results_df.to_csv(os.path.join(summary_dir, "synonym_best_models.csv"))


def model_performance_bar_graphs(synonym_results_df: pandas.DataFrame, window_radius: int, distance_type: DistanceType):

    figures_dir = Preferences.figures_dir
    seaborn.set_style("ticks")

    filtered_df: pandas.DataFrame = synonym_results_df.copy()
    filtered_df = filtered_df[filtered_df["radius"] == window_radius]
    filtered_df = filtered_df[filtered_df["distance"] == distance_type.name]

    # Don't want to show PPMI (10000)
    filtered_df = filtered_df[filtered_df["model"] != "PPMI (10000)"]

    # Model name doesn't need to include corpus or distance, since those are fixed
    filtered_df["model_name"] = filtered_df.apply(
        lambda r:
        f"{r['model']} {r['embedding_size']}"
        if not numpy.math.isnan(r['embedding_size'])
        else f"{r['model']}",
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

    ytick_labels = grid.axes[0][0].get_yticklabels()
    grid.set_yticklabels(['{:3.0f}%'.format(float(label.get_text()) * 100) for label in ytick_labels])

    # Plot the bars
    plot = grid.map(seaborn.barplot, "model_name", "score", order=[
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
    grid.set_titles(col_template='{col_name}', row_template="{row_name}")

    # Plot the chance line
    grid.map(pyplot.axhline, y=0.25, linestyle="solid", color="xkcd:bright red")

    grid.set_ylabels("Score")

    pyplot.subplots_adjust(top=0.92)
    grid.fig.suptitle(f"Model scores for radius {window_radius} using {distance_type.name} distance")

    figure_name = f"synonym r={window_radius} {distance_type.name}.png"

    # I don't know why PyCharm doesn't find this... it works...
    # noinspection PyUnresolvedReferences
    plot.savefig(os.path.join(figures_dir, figure_name), dpi=300)


def figures_score_vs_radius(regression_results_df: pandas.DataFrame, test_name: str):
    figures_dir = Preferences.figures_dir
    for distance in [d.name for d in DistanceType]:
        for corpus in ["BNC", "BBC", "UKWAC"]:
            filtered_df: pandas.DataFrame = regression_results_df.copy()
            filtered_df = filtered_df[filtered_df["corpus"] == corpus]
            filtered_df = filtered_df[filtered_df["distance"] == distance]
            filtered_df = filtered_df[filtered_df["test_name"] == test_name]

            filtered_df = filtered_df.sort_values(by=["model_name", "radius"])
            filtered_df = filtered_df.reset_index(drop=True)

            filtered_df = filtered_df[[
                "model_name",
                "radius",
                "score"]]

            plot = seaborn.factorplot(data=filtered_df,
                                      x="radius", y="score",
                                      hue="model_name",
                                      size=7, aspect=1.8,
                                      legend=False)

            plot.set(ylim=(0, 1))

            # Format yticks as percentages
            vals = plot.ax.get_yticks()
            plot.ax.set_yticklabels(['{:3.0f}%'.format(x * 100) for x in vals])

            # Put the legend out of the figure
            # resize figure box to -> put the legend out of the figure
            plot_box = plot.ax.get_position()  # get position of figure
            plot.ax.set_position([plot_box.x0, plot_box.y0, plot_box.width * 0.75, plot_box.height])  # resize position

            # Put a legend to the right side
            plot.ax.legend(loc='center right', bbox_to_anchor=(1.35, 0.5), ncol=1)

            figure_name = f"synonym {test_name} {corpus} {distance}.png"

            plot.savefig(os.path.join(figures_dir, figure_name))


def figures_embedding_size(regression_results_df: pandas.DataFrame, test_name: str):

    figures_dir = Preferences.figures_dir
    for distance in [d.name for d in DistanceType]:
        filtered_df: pandas.DataFrame = regression_results_df.copy()
        filtered_df = filtered_df[filtered_df["distance"] == distance]
        filtered_df = filtered_df[filtered_df["test_name"] == test_name]

        # Remove count models by dropping rows with nan in embedding_size column
        filtered_df = filtered_df.dropna()

        filtered_df = filtered_df[[
            "corpus",
            "model",
            "embedding_size",
            "radius",
            "score"]]

        filtered_df = filtered_df.sort_values(by=["corpus", "model", "embedding_size", "radius"])
        filtered_df = filtered_df.reset_index(drop=True)

        seaborn.set_style("ticks")
        seaborn.set_context(context="paper", font_scale=1)
        grid = seaborn.FacetGrid(
            filtered_df,
            row="radius", col="corpus", hue="model",
            margin_titles=True,
            size=2,
            ylim=(0, 1),
            legend_out=True
        )

        # Chance line
        grid.map(pyplot.axhline, y=0.25, ls=":", c=".5", label="")

        grid.map(pyplot.plot, "embedding_size", "score", marker="o")

        grid.set(
            xticks=Preferences.predict_embedding_sizes,
        )

        ytick_labels = grid.axes[0][0].get_yticklabels()
        grid.set_yticklabels(['{:3.0f}%'.format(float(label.get_text()) * 100) for label in ytick_labels])

        grid.set_xlabels("Embedding size")
        grid.set_ylabels("Score")

        grid.add_legend(title="Model", bbox_to_anchor=(1, 1))
        # pyplot.legend(loc="lower center")

        # Title
        title = f"{test_name} ({distance})"
        pyplot.subplots_adjust(top=0.92)
        grid.fig.suptitle(title)

        figure_name = f"synonym embedding_size {test_name} {distance}.png"

        grid.savefig(os.path.join(figures_dir, figure_name), dpi=300)


# TODO: this is madness, this should be a pickled ReportCard which should have the ability to export a csv or a dataframe
# TODO: Also, ReportCard is basically just a dataframe, and should be replaced by one.
def load_data() -> pandas.DataFrame:
    """
    Load a pandas.DataFrame from a collection of CSV fragments.
    """
    results_dir = Preferences.synonym_results_dir
    separator = ","

    def percent_to_float(percent: str) -> float:
        return float(percent.strip("%")) / 100

    header_filename = os.path.join(results_dir, " header.csv")
    data_filenames = glob(os.path.join(results_dir, "*.csv"))
    data_filenames.remove(header_filename)

    with open(os.path.join(results_dir, " header.csv"), mode="r", encoding="utf-8") as header_file:
        column_names = header_file.read().strip().split(separator)

    results_df = pandas.DataFrame(columns=column_names)

    for data_filename in data_filenames:
        partial_df = pandas.read_csv(data_filename, sep=separator, names=column_names,
                                     converters={'Score': percent_to_float})
        results_df = results_df.append(partial_df, ignore_index=True)

    return results_df


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
