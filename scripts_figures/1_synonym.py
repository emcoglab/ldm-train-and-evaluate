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

import matplotlib.pyplot as pplot

from ..core.utils.logging import log_message, date_format
from ..core.utils.maths import DistanceType
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def ensure_column_safety(df: pandas.DataFrame) -> pandas.DataFrame:
    return df.rename(columns=lambda col_name: col_name.replace(" ", "_").lower())


def main():
    figures_dir = Preferences.figures_dir

    df = load_data()

    df = ensure_column_safety(df)

    df["model_name"] = df.apply(lambda r:
                                f"{r['corpus']} {r['distance']} {r['model']} {r['embedding_size']}"
                                if r['embedding_size'] != numpy.nan
                                else f"{r['corpus']} {r['distance']} {r['model']}",
                                axis=1)

    distance = DistanceType.cosine.name
    corpus = "BNC"
    model = "CBOW"
    embedding_size = 500
    test_name = "TOEFL"

    fdf: pandas.DataFrame = df.copy()
    fdf = fdf[fdf["corpus"] == corpus]
    fdf = fdf[fdf["distance"] == distance]
    fdf = fdf[fdf["model"] == model]
    fdf = fdf[fdf["embedding_size"] == embedding_size]
    fdf = fdf[fdf["test_name"] == test_name]

    fdf = fdf.sort_values(by=["model_name", "radius"])
    fdf = fdf.reset_index(drop=True)

    fdf = fdf[[
        "model_name",
        "radius",
        "score"]]

    # plot = seaborn.factorplot(data=fdf, x='radius', y="score")
    #
    # plot.savefig(os.path.join(figures_dir, "test.png"))

    ax = fdf.plot(x="radius", y="score", ylim=[0, 1])

    # Format as percent
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.2f}%'.format(x * 100) for x in vals])

    # Save figure
    pplot.savefig(os.path.join(figures_dir, "test.png"))

    pass


def load_data() -> pandas.DataFrame:
    """
    Load a pandas.DataFrame from a collection of CSV fragments.
    """
    results_dir = Preferences.synonym_results_dir
    separator = ","

    def percent_to_float(percent: str) -> float:
        return float(percent.strip("%"))/100

    header_filename = os.path.join(results_dir, " header.csv")
    data_filenames = glob(os.path.join(results_dir, "*.csv"))
    data_filenames.remove(header_filename)

    with open(os.path.join(results_dir, " header.csv"), mode="r", encoding="utf-8") as header_file:
        column_names = header_file.read().strip().split(separator)

    data = pandas.DataFrame(columns=column_names)

    for data_filename in data_filenames:
        partial_df = pandas.read_csv(data_filename, sep=separator, names=column_names, converters={'Score': percent_to_float})
        data = data.append(partial_df, ignore_index=True)

    return data


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
