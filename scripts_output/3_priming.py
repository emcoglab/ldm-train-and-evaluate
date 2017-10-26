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

DV_NAMES = []


def ensure_column_safety(df: pandas.DataFrame) -> pandas.DataFrame:
    return df.rename(columns=lambda col_name: col_name.replace(" ", "_").lower())


# TODO: essentially duplicated code
def main():

    spp_results_df = load_data()
    spp_results_df = ensure_column_safety(spp_results_df)

    summary_tables(spp_results_df)


def summary_tables(similarity_results: pandas.DataFrame):
    summary_dir = Preferences.summary_dir

    for correlation_type in [c.name for c in CorrelationType]:

        results_df = pandas.DataFrame()

        for test_name in DV_NAMES:

            filtered_df: pandas.DataFrame = similarity_results.copy()
            filtered_df = filtered_df[filtered_df["test_name"] == test_name]
            filtered_df = filtered_df[filtered_df["correlation_type"] == correlation_type]

            # min because correlations are negative
            best_correlation = filtered_df["correlation"].min()

            best_models_df = filtered_df[filtered_df["correlation"] == best_correlation]

            results_df = results_df.append(best_models_df)

        results_df = results_df.reset_index(drop=True)

        results_df.to_csv(os.path.join(summary_dir, f"similarity_best_models_{correlation_type.lower()}.csv"))


def load_data() -> pandas.DataFrame:
    """
    Load a pandas.DataFrame from a collection of CSV fragments.
    """
    results_dir = Preferences.spp_results_dir
    separator = ","

    with open(os.path.join(results_dir, "regression.csv"), mode="r", encoding="utf-8") as regression_file:
        return pandas.read_csv(regression_file, sep=separator, header=True)


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
