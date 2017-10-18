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
from ..core.utils.maths import DistanceType
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def ensure_column_safety(df: pandas.DataFrame) -> pandas.DataFrame:
    return df.rename(columns=lambda col_name: col_name.replace(" ", "_").lower())


def main():
    figures_dir = Preferences.figures_dir

    dataframe = load_data()

    dataframe = ensure_column_safety(dataframe)

    dataframe["model_name"] = dataframe.apply(lambda r:
                                              f"{r['corpus']} {r['distance']} {r['model']} {r['embedding_size']}"
                                              if not numpy.math.isnan(r['embedding_size'])
                                              else f"{r['corpus']} {r['distance']} {r['model']}",
                                              axis=1)

    for test_name in ["Simlex-999", "WordSim-353 similarity", "WordSim-353 relatedness", "MEN"]:
        for distance in [d.name for d in DistanceType]:
            for corpus in ["BNC", "BBC", "UKWAC"]:

                figure_name = f"similarity {test_name} {corpus} {distance}.png"

                filtered_dataframe: pandas.DataFrame = dataframe.copy()
                filtered_dataframe = filtered_dataframe[filtered_dataframe["corpus"] == corpus]
                filtered_dataframe = filtered_dataframe[filtered_dataframe["distance"] == distance]
                filtered_dataframe = filtered_dataframe[filtered_dataframe["test_name"] == test_name]

                filtered_dataframe = filtered_dataframe.sort_values(by=["model_name", "radius"])
                filtered_dataframe = filtered_dataframe.reset_index(drop=True)

                filtered_dataframe = filtered_dataframe[[
                    "model_name",
                    "radius",
                    "correlation"]]

                plot = seaborn.factorplot(data=filtered_dataframe, x="radius", y="correlation", hue="model_name")

                plot.set(ylim=(0, 1))

                # Format yticks as percentages
                vals = plot.ax.get_yticks()
                plot.ax.set_yticklabels(['{:3f}%'.format(x * 100) for x in vals])

                plot.savefig(os.path.join(figures_dir, figure_name))


# TODO: essentially duplicated code
def load_data() -> pandas.DataFrame:
    """
    Load a pandas.DataFrame from a collection of CSV fragments.
    """
    results_dir = Preferences.similarity_results_dir
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
                                     converters={'Correlation': percent_to_float})
        data = data.append(partial_df, ignore_index=True)

    return data


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
