"""
===========================
Evaluate using priming data: regress predictors against SPP data.
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
import sys
import os

from typing import List

import pandas
import statsmodels.formula.api as sm

from ..core.model.base import VectorSemanticModel
from ..core.utils.maths import DistanceType
from ..core.corpus.distribution import FreqDist
from ..core.utils.indexing import TokenIndexDictionary
from ..core.model.count import LogNgramModel, ConditionalProbabilityModel, ProbabilityRatioModel, PPMIModel
from ..core.model.predict import SkipGramModel, CbowModel
from ..preferences.preferences import Preferences
from ..core.evaluation.priming import SppData, PrimingRegressionResult
from ..core.utils.logging import log_message, date_format


logger = logging.getLogger(__name__)


def export_csv(results: List[PrimingRegressionResult]):
    results_path = os.path.join(Preferences.spp_results_dir, "regression.csv")

    separator = ","
    with open(results_path, mode="w", encoding="utf-8") as results_file:
        # Print header
        results_file.write(separator.join(PrimingRegressionResult.headings()) + "\n")
        # Print results
        for result in results:
            results_file.write(separator.join(result.fields) + '\n')


def fit_all_models(all_data: pandas.DataFrame, dependent_variable_names: List[str], baseline_variable_names: List[str]):

    results : List[PrimingRegressionResult] = []

    for corpus_metadata in Preferences.source_corpus_metas:

        token_index = TokenIndexDictionary.load(corpus_metadata.index_path)
        freq_dist = FreqDist.load(corpus_metadata.freq_dist_path)

        for window_radius in Preferences.window_radii:

            # Count models

            count_models = [
                LogNgramModel(corpus_metadata, window_radius, token_index),
                ConditionalProbabilityModel(corpus_metadata, window_radius, token_index, freq_dist),
                ProbabilityRatioModel(corpus_metadata, window_radius, token_index, freq_dist),
                PPMIModel(corpus_metadata, window_radius, token_index, freq_dist)
            ]

            for model in count_models:
                for distance_type in DistanceType:
                    for dv_name in dependent_variable_names:
                        result = run_regression(all_data, distance_type, dv_name, model, baseline_variable_names)
                        results.append(result)

            # Predict models

            for embedding_size in Preferences.predict_embedding_sizes:

                predict_models = [
                    SkipGramModel(corpus_metadata, window_radius, embedding_size),
                    CbowModel(corpus_metadata, window_radius, embedding_size)
                ]

                for model in predict_models:
                    for distance_type in DistanceType:
                        for dv_name in dependent_variable_names:
                            result = run_regression(all_data, distance_type, dv_name, model, baseline_variable_names)
                            results.append(result)
    return results


def run_regression(all_data, distance_type, dv_name, model: VectorSemanticModel, baseline_variable_names: List[str]):

    model_predictor_name = SppData.predictor_name_for_model(model, distance_type)

    logger.info(f"Running {dv_name} regressions for model {model_predictor_name}")

    # Formulae
    baseline_formula = f"{dv_name} ~ {' + '.join(baseline_variable_names)}"
    model_formula = f"{baseline_formula} + {model_predictor_name}"

    baseline_regression = sm.ols(
        formula=baseline_formula,
        data=all_data).fit()
    model_regression = sm.ols(
        formula=model_formula,
        data=all_data).fit()

    return PrimingRegressionResult(
        dv_name,
        model,
        distance_type,
        baseline_regression.rsquared,
        model_regression.rsquared)


def main():

    spp_data: SppData = SppData()

    # Get only the first associate prime–target pairs
    first_assoc_prime_data = spp_data.dataframe.query('PrimeType == "first_associate"')

    # Compute all models for non-priming data

    dependent_variable_names = [
        "LDT_200ms_Z",
        "LDT_200ms_Acc",
        "LDT_1200ms_Z",
        "LDT_1200ms_Acc",
        "LDT_mean_Z",
        "LDT_mean_Acc",
        "NT_200ms_Z",
        "NT_200ms_Acc",
        "NT_1200ms_Z",
        "NT_1200ms_Acc",
        "NT_mean_Z",
        "NT_mean_Acc"
    ]

    baseline_variable_names = [
        "TargetLength",
        "PrimeLength",
        "elex_prime_LgSUBTLWF",
        "elex_target_LgSUBTLWF",
        "elex_prime_OLD",
        "elex_target_OLD",
        "elex_prime_PLD",
        "elex_target_PLD",
        "elex_prime_NSyll",
        "elex_target_NSyll",
        "PrimeTarget_OrthLD"
    ]

    results = fit_all_models(first_assoc_prime_data, dependent_variable_names, baseline_variable_names)

    # Compute all models for priming data

    dependent_variable_priming_names = [
        "LDT_200ms_Z_Priming",
        "LDT_200ms_Acc_Priming",
        "LDT_1200ms_Z_Priming",
        "LDT_1200ms_Acc_Priming",
        "LDT_mean_Z_Priming",
        "LDT_mean_Acc_Priming",
        "NT_200ms_Z_Priming",
        "NT_200ms_Acc_Priming",
        "NT_1200ms_Z_Priming",
        "NT_1200ms_Acc_Priming",
        "NT_mean_Z_Priming",
        "NT_mean_Acc_Priming"
    ]

    baseline_variable_priming_names = [
        "PrimeTarget_OrthLD"
    ]

    priming_results = fit_all_models(first_assoc_prime_data, dependent_variable_priming_names, baseline_variable_priming_names)

    results.extend(priming_results)

    export_csv(results)


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
