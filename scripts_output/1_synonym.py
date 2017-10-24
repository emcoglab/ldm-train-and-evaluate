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


def ensure_column_safety(df: pandas.DataFrame) -> pandas.DataFrame:
    return df.rename(columns=lambda col_name: col_name.replace(" ", "_").lower())


def main():
    dataframe = load_data()
    dataframe = ensure_column_safety(dataframe)

    dataframe["model_name"] = dataframe.apply(lambda r:
                                              f"{r['corpus']} {r['distance']} {r['model']} {r['embedding_size']}"
                                              if not numpy.math.isnan(r['embedding_size'])
                                              else f"{r['corpus']} {r['distance']} {r['model']}",
                                              axis=1)

    for test_name in ["TOEFL", "ESL", "LBM's new MCQ"]:
        # figures_score_vs_radius(dataframe, test_name)
        figures_embedding_size(dataframe, test_name)
        summary_tables(dataframe, test_name)


def summary_tables(dataframe: pandas.DataFrame, test_name: str):
    summary_dir = Preferences.summary_dir
    filtered_dataframe: pandas.DataFrame = dataframe.copy()


def figures_score_vs_radius(dataframe: pandas.DataFrame, test_name: str):
    figures_dir = Preferences.figures_dir
    for distance in [d.name for d in DistanceType]:
        for corpus in ["BNC", "BBC", "UKWAC"]:
            filtered_dataframe: pandas.DataFrame = dataframe.copy()
            filtered_dataframe = filtered_dataframe[filtered_dataframe["corpus"] == corpus]
            filtered_dataframe = filtered_dataframe[filtered_dataframe["distance"] == distance]
            filtered_dataframe = filtered_dataframe[filtered_dataframe["test_name"] == test_name]

            filtered_dataframe = filtered_dataframe.sort_values(by=["model_name", "radius"])
            filtered_dataframe = filtered_dataframe.reset_index(drop=True)

            filtered_dataframe = filtered_dataframe[[
                "model_name",
                "radius",
                "score"]]

            plot = seaborn.factorplot(data=filtered_dataframe,
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


def figures_embedding_size(dataframe: pandas.DataFrame, test_name: str):

    figures_dir = Preferences.figures_dir
    for distance in [d.name for d in DistanceType]:
        filtered_dataframe: pandas.DataFrame = dataframe.copy()
        filtered_dataframe = filtered_dataframe[filtered_dataframe["distance"] == distance]
        filtered_dataframe = filtered_dataframe[filtered_dataframe["test_name"] == test_name]

        # Remove count models by dropping rows with nan in embedding_size column
        filtered_dataframe = filtered_dataframe.dropna()

        filtered_dataframe = filtered_dataframe[[
            "corpus",
            "model",
            "embedding_size",
            "radius",
            "score"]]

        filtered_dataframe = filtered_dataframe.sort_values(by=["corpus", "model", "embedding_size", "radius"])
        filtered_dataframe = filtered_dataframe.reset_index(drop=True)

        seaborn.set_style("ticks")
        seaborn.set_context(context="paper", font_scale=1)
        grid = seaborn.FacetGrid(
            filtered_dataframe,
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

    data = pandas.DataFrame(columns=column_names)

    for data_filename in data_filenames:
        partial_df = pandas.read_csv(data_filename, sep=separator, names=column_names,
                                     converters={'Score': percent_to_float})
        data = data.append(partial_df, ignore_index=True)

    return data


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
