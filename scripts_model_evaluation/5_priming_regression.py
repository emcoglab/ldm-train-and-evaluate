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

import statsmodels.formula.api as sm

from ..core.utils.maths import DistanceType
from ..core.corpus.distribution import FreqDist
from ..core.utils.indexing import TokenIndexDictionary
from ..core.model.count import LogNgramModel, ConditionalProbabilityModel, ProbabilityRatioModel, PPMIModel
from ..core.model.predict import SkipGramModel, CbowModel
from ..preferences.preferences import Preferences
from ..core.evaluation.priming import SppData, BaselineRegression, ModelRegression
from ..core.utils.logging import log_message, date_format


logger = logging.getLogger(__name__)


def main():

    spp_data: SppData = SppData()

    first_assoc_prime_data = spp_data.dataframe.where(
        # TODO!
    )

    # Compute all models for non-priming data

    dependent_variable_names = [
        "LDT_200ms_Z",
        "LDT_200ms_Acc",
        "LDT_1200ms_Z",
        "LDT_1200ms_Acc"
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
        "LDT_1200ms_Acc_Priming"
    ]

    baseline_variable_priming_names = [
        "PrimeTarget_OrthLD"
    ]

    priming_results = fit_all_models(first_assoc_prime_data, dependent_variable_priming_names, baseline_variable_priming_names)

    results.extend(priming_results)

    for result in results:
        logger.info(f"{result.name}:\t{result.rsquared}")


def fit_all_models(all_data, dependent_variable_names, baseline_variable_names):

    results = []

    # Baseline models
    for dv_name in dependent_variable_names:
        # Predictor variables
        predictor_formula = ' + '.join(baseline_variable_names)

        results.append(BaselineRegression(
            dv_name=dv_name,
            result=sm.ols(
                formula=f"{dv_name} ~ {predictor_formula}",
                data=all_data).fit()))

    # Compute full models for non-priming data
    for corpus_metadata in Preferences.source_corpus_metas:

        token_index = TokenIndexDictionary.load(corpus_metadata.index_path)
        freq_dist = FreqDist.load(corpus_metadata.freq_dist_path)

        for window_radius in Preferences.window_radii:

            count_models = [
                LogNgramModel(corpus_metadata, window_radius, token_index),
                ConditionalProbabilityModel(corpus_metadata, window_radius, token_index, freq_dist),
                ProbabilityRatioModel(corpus_metadata, window_radius, token_index, freq_dist),
                PPMIModel(corpus_metadata, window_radius, token_index, freq_dist)
            ]

            for model in count_models:
                for distance_type in DistanceType:

                    model_predictor_name = SppData.predictor_name_for_model(model, distance_type)

                    predictor_formula = ' + '.join(baseline_variable_names + [model_predictor_name])

                    for dv_name in dependent_variable_names:
                        results += ModelRegression(
                            dv_name=dv_name,
                            model=model,
                            distance_type=distance_type,
                            result=sm.ols(
                                formula=f"{dv_name} ~ {predictor_formula}",
                                data=all_data).fit())

            for embedding_size in Preferences.predict_embedding_sizes:

                predict_models = [
                    SkipGramModel(corpus_metadata, window_radius, embedding_size),
                    CbowModel(corpus_metadata, window_radius, embedding_size)
                ]

                for model in predict_models:
                    for distance_type in DistanceType:

                        model_predictor_name = SppData.predictor_name_for_model(model, distance_type)

                        for dv_name in dependent_variable_names:
                            results.append(ModelRegression(
                                dv_name=dv_name,
                                model=model,
                                distance_type=distance_type,
                                result=sm.ols(
                                    formula=f"{dv_name} ~ {model_predictor_name}",
                                    data=all_data).fit()))
    return results


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
