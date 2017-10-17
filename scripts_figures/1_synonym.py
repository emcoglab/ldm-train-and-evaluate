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

import matplotlib.pyplot as plot

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

    df["model_name"] = df.apply(lambda r: f"{r['corpus']} {r['distance']} {r['model']} {r['embedding_size']}", axis=1)

    distance = DistanceType.cosine.name
    corpus = "BNC"
    model= "CBOW"
    embedding_size = numpy.nan
    test_name = "TOEFL"

    fdf = df.query(
        f"corpus == '{corpus}' & "
        f"distance == '{distance}' & "
        # f"model == '{model}' & "
        # f"embedding_size == {embedding_size} & "
        f"test_name == '{test_name}'"
    )

    fdf = fdf.sort_values(by=["model_name", "radius"]).reindex()

    fdf = fdf[["model_name", "radius", "score"]]

    seaborn.factorplot(data=fdf, x='radius', y="score", hue="model_name")

    pass


def load_data() -> pandas.DataFrame:
    """
    Load a pandas.DataFrame from a collection of CSV fragments.
    """
    results_dir = Preferences.synonym_results_dir
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
