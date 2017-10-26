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

from ..core.utils.logging import log_message, date_format
from ..core.utils.maths import CorrelationType
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

    summary_tables(spp_results_df)


def summary_tables(spp_results: pandas.DataFrame):
    summary_dir = Preferences.summary_dir

    results_df = pandas.DataFrame()

    for dv_name in DV_NAMES:

        filtered_df: pandas.DataFrame = spp_results.copy()
        filtered_df = filtered_df[filtered_df["dependent_variable"] == dv_name]

        # min because correlations are negative
        best_r2 = filtered_df["r-squared_increase"].max()

        best_models_df = filtered_df[filtered_df["r-squared_increase"] == best_r2]

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
