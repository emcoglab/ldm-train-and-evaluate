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

from glob import glob

import numpy
import pandas
import seaborn

from ..core.utils.logging import log_message, date_format
from ..core.utils.maths import DistanceType, CorrelationType
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)

TEST_NAMES = ["Simlex-999", "WordSim-353 similarity", "WordSim-353 relatedness", "MEN"]


def ensure_column_safety(df: pandas.DataFrame) -> pandas.DataFrame:
    return df.rename(columns=lambda col_name: col_name.replace(" ", "_").lower())


# TODO: essentially duplicated code
def main():

    similarity_results = load_data()
    similarity_results = ensure_column_safety(similarity_results)

    similarity_results["model"] = similarity_results.apply(
        lambda r:
        f"{r['corpus']} {r['distance_type']} {r['model_type']} {r['embedding_size']}"
        if not numpy.math.isnan(r['embedding_size'])
        else f"{r['corpus']} {r['distance_type']} {r['model_type']}",
        axis=1
    )

    for test_name in TEST_NAMES:
        figures_score_vs_radius(similarity_results, test_name)

    summary_tables(similarity_results)


def summary_tables(similarity_results: pandas.DataFrame):
    summary_dir = Preferences.summary_dir

    for correlation_type in [c.name for c in CorrelationType]:

        results_df = pandas.DataFrame()

        for test_name in TEST_NAMES:

            filtered_df: pandas.DataFrame = similarity_results.copy()
            filtered_df = filtered_df[filtered_df["test_name"] == test_name]
            filtered_df = filtered_df[filtered_df["correlation_type"] == correlation_type]

            # min because correlations are negative
            best_correlation = filtered_df["correlation"].min()

            best_models_df = filtered_df[filtered_df["correlation"] == best_correlation]

            results_df = results_df.append(best_models_df)

        results_df = results_df.reset_index(drop=True)

        results_df.to_csv(os.path.join(summary_dir, f"similarity_best_models_{correlation_type.lower()}.csv"))


def figures_score_vs_radius(similarity_results, test_name):
    figures_dir = Preferences.figures_dir
    for distance in [d.name for d in DistanceType]:
        for corpus in ["BNC", "BBC", "UKWAC"]:
            figure_name = f"similarity {test_name} {corpus} {distance}.png"

            filtered_dataframe: pandas.DataFrame = similarity_results.copy()
            filtered_dataframe = filtered_dataframe[filtered_dataframe["corpus"] == corpus]
            filtered_dataframe = filtered_dataframe[filtered_dataframe["distance_type"] == distance]
            filtered_dataframe = filtered_dataframe[filtered_dataframe["test_name"] == test_name]

            filtered_dataframe = filtered_dataframe.sort_values(by=["model", "window_radius"])
            filtered_dataframe = filtered_dataframe.reset_index(drop=True)

            filtered_dataframe = filtered_dataframe[[
                "model",
                "window_radius",
                "correlation"
            ]]

            plot = seaborn.factorplot(data=filtered_dataframe,
                                      x="window_radius", y="correlation",
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


def load_data() -> pandas.DataFrame:
    """
    Load a pandas.DataFrame from a collection of CSV fragments.
    """
    results_dir = Preferences.similarity_results_dir
    separator = ","

    header_filename = os.path.join(results_dir, " header.csv")
    data_filenames = glob(os.path.join(results_dir, "*.csv"))
    data_filenames.remove(header_filename)

    with open(os.path.join(results_dir, " header.csv"), mode="r", encoding="utf-8") as header_file:
        column_names = header_file.read().strip().split(separator)

    data = pandas.DataFrame(columns=column_names)

    for data_filename in data_filenames:
        partial_df = pandas.read_csv(data_filename, sep=separator, names=column_names)
        data = data.append(partial_df, ignore_index=True)

    return data


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
