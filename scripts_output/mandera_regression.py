"""
===========================
Re-running regressions with Mandera et al.’s proximity values, to check for discrepancy.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2018
---------------------------
"""

import logging
from os import path

import numpy
import sys

from pandas import read_csv, DataFrame
import statsmodels.formula.api as sm


from ..core.utils.logging import log_message, date_format
from ..core.evaluation.regression import SppData
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)

MANDERA_COL_NAME = 'Mandera_cosine_cbow300_r6_ukwacsubtitles'


def main():
    spp_data = SppData(save_progress=False)
    add_mandera_predictor(spp_data)
    spp_data.export_csv_first_associate_only(path=path.join(Preferences.spp_results_dir, "mandera_predictors_first_associate_only.csv"))
    results_df = run_mandera_regression(spp_data)
    results_df.to_csv(Preferences.mandera_results_csv, header=True, index=False, encoding="utf-8")


def add_mandera_predictor(spp_data):
    mandera_distances = read_csv(Preferences.mandera_distances_csv, header=0, encoding="utf-8")
    mandera_distances.rename(columns={
        "word_1": "PrimeWord",
        "word_2": "TargetWord",
        "distance": MANDERA_COL_NAME
    }, inplace=True)
    spp_data.add_word_pair_keyed_predictor(mandera_distances, merge_on=["PrimeWord", "TargetWord"])


def run_mandera_regression(spp_data):
    first_assoc_prime_data = spp_data.dataframe.query('PrimeType == "first_associate"')

    dependent_variable_names = [
        "LDT_200ms_Z",
        "LDT_200ms_Acc",
        "LDT_1200ms_Z",
        "LDT_1200ms_Acc",
        "NT_200ms_Z",
        "NT_200ms_Acc",
        "NT_1200ms_Z",
        "NT_1200ms_Acc"
    ]

    baseline_variable_names = [
        "TargetLength",
        "elex_target_LgSUBTLWF",
        "elex_target_OLD",
        "elex_target_PLD",
        "elex_target_NSyll",
        "PrimeTarget_OrthLD"
    ]

    results = []
    for dv_name in dependent_variable_names:
        # drop rows with missing data in any relevant column
        regression_data = first_assoc_prime_data[[dv_name] + baseline_variable_names + [MANDERA_COL_NAME]].dropna(how="any")

        baseline_formula = f"{dv_name} ~ {'+'.join(baseline_variable_names)}"
        mandera_formula = f"{baseline_formula} + {MANDERA_COL_NAME}"

        baseline_regression = sm.ols(
            formula=baseline_formula,
            data=regression_data).fit()
        mandera_regression = sm.ols(
            formula=mandera_formula,
            data=regression_data).fit()

        results.append([
            dv_name,
            baseline_regression.rsquared,
            mandera_regression.rsquared,
            mandera_regression.rsquared - baseline_regression.rsquared,
            baseline_regression.bic,
            mandera_regression.bic,
            numpy.exp((baseline_regression.bic - mandera_regression.bic) / 2),
            ((baseline_regression.bic - mandera_regression.bic) / 2) * numpy.log10(numpy.exp(1)),
            mandera_regression.tvalues[MANDERA_COL_NAME],
            mandera_regression.pvalues[MANDERA_COL_NAME],
            mandera_regression.params[MANDERA_COL_NAME],
            mandera_regression.df_resid
        ])

    return DataFrame(results, columns=[
        "Dependent variable",
        "Baseline R-squared",
        "Mandera R-squared",
        "R-squared increase",
        "Baseline BIC",
        "Mandera BIC",
        "B10 approx",
        "Log10 B10 approx",
        "t",
        "p",
        "beta",
        "df"
    ])


if __name__ == '__main__':
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
