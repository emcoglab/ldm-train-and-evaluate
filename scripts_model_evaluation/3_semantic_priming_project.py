"""
===========================
Evaluate using priming data.
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

import os
import logging
import sys

from typing import Set, List

import pandas
import statsmodels.formula.api as sm

from ..core.model.base import VectorSemanticModel
from ..core.utils.maths import DistanceType, levenshtein_distance
from ..core.evaluation.priming import SppData, SppRegressionResult
from ..core.model.count import LogNgramModel, ConditionalProbabilityModel, ProbabilityRatioModel, PPMIModel
from ..core.model.predict import SkipGramModel, CbowModel
from ..core.utils.indexing import TokenIndexDictionary
from ..core.corpus.distribution import FreqDist
from ..core.utils.logging import log_message, date_format
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def export_csv(results: List[SppRegressionResult]):
    results_path = os.path.join(Preferences.spp_results_dir, "regression.csv")

    separator = ","
    with open(results_path, mode="w", encoding="utf-8") as results_file:
        # Print header
        results_file.write(separator.join(SppRegressionResult.headings()) + "\n")
        # Print results
        for result in results:
            results_file.write(separator.join(result.fields) + '\n')


def fit_all_models(all_data: pandas.DataFrame, dependent_variable_names: List[str], baseline_variable_names: List[str]):

    results : List[SppRegressionResult] = []

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


# TODO: there is a lot of shared code between this function and fit_all_models
def fit_all_priming_models(all_data: pandas.DataFrame, dependent_variable_names: List[str], baseline_variable_names: List[str]):

    results : List[SppRegressionResult] = []

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
                        result = run_priming_regression(all_data, distance_type, dv_name, model, baseline_variable_names)
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

    return SppRegressionResult(
        dv_name,
        model,
        distance_type,
        baseline_regression.rsquared,
        model_regression.rsquared)


# TODO: there is a lot of shared code between this and run_regression
def run_priming_regression(all_data, distance_type, dv_name, model: VectorSemanticModel, baseline_variable_names: List[str]):

    model_predictor_name = SppData.priming_predictor_name_for_model(model, distance_type)

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

    return SppRegressionResult(
        dv_name,
        model,
        distance_type,
        baseline_regression.rsquared,
        model_regression.rsquared)


def main():
    spp_data = SppData()

    save_wordlist(spp_data.vocabulary)

    for corpus_metadata in Preferences.source_corpus_metas:

        token_index = TokenIndexDictionary.load(corpus_metadata.index_path)
        freq_dist = FreqDist.load(corpus_metadata.freq_dist_path)

        for window_radius in Preferences.window_radii:

            # COUNT MODELS

            count_models = [
                LogNgramModel(corpus_metadata, window_radius, token_index),
                ConditionalProbabilityModel(corpus_metadata, window_radius, token_index, freq_dist),
                ProbabilityRatioModel(corpus_metadata, window_radius, token_index, freq_dist),
                PPMIModel(corpus_metadata, window_radius, token_index, freq_dist)
            ]

            for model in count_models:
                add_predictors_for_model(model, spp_data)

            # PREDICT MODELS

            for embedding_size in Preferences.predict_embedding_sizes:

                predict_models = [
                    SkipGramModel(corpus_metadata, window_radius, embedding_size),
                    CbowModel(corpus_metadata, window_radius, embedding_size)
                ]

                for model in predict_models:
                    add_predictors_for_model(model, spp_data)

    add_elexicon_predictors(spp_data)

    do_the_regression(spp_data)

    spp_data.export_csv()


def do_the_regression(spp_data: SppData):
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
        # "PrimeLength",
        "TargetLength",
        # "elex_prime_LgSUBTLWF",
        "elex_target_LgSUBTLWF",
        # "elex_prime_OLD",
        "elex_target_OLD",
        # "elex_prime_PLD",
        "elex_target_PLD",
        # "elex_prime_NSyll",
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
        "PrimeTarget_OrthLD_Priming"
    ]

    priming_results = fit_all_priming_models(first_assoc_prime_data, dependent_variable_priming_names,
                                             baseline_variable_priming_names)

    results.extend(priming_results)


def add_elexicon_predictors(spp_data: SppData):
    # Add Elexicon predictors

    elexicon_dataframe: pandas.DataFrame = pandas.read_csv(Preferences.spp_elexicon_csv, header=0, encoding="utf-8")

    # Make sure the words are lowercase, as we'll use them as merging keys
    elexicon_dataframe['Word'] = elexicon_dataframe['Word'].str.lower()

    predictors_to_add = [
        "LgSUBTLWF",
        "OLD",
        "PLD",
        "NSyll"
    ]

    for predictor_name in predictors_to_add:
        add_elexicon_predictor(spp_data, elexicon_dataframe, predictor_name, prime_or_target="Prime")
        add_elexicon_predictor(spp_data, elexicon_dataframe, predictor_name, prime_or_target="Target")

    # Add prime-target Levenschtein distance

    # Add Levenshtein distance column to data frame
    levenshtein_column_name = "PrimeTarget_OrthLD"
    if spp_data.predictor_exists_with_name(levenshtein_column_name):
        logger.info("Levenshtein-distance predictor already added to SPP data.")
    else:
        logger.info("Adding Levenshtein-distance predictor to SPP data.")

        word_columns = ["PrimeWord", "TargetWord"]
        word_pairs = spp_data.dataframe[word_columns].copy()
        word_pairs[levenshtein_column_name] = word_pairs[word_columns].apply(levenshtein_distance_local, axis=1)

        spp_data.add_word_pair_keyed_predictor(word_pairs, merge_on=word_columns)

    # Add Levenshtein priming distance column to data frame
    priming_levenshtein_column_name = "PrimeTarget_OrthLD_Priming"
    if spp_data.predictor_exists_with_name(priming_levenshtein_column_name):
        logger.info("Levenshtein-distance priming predictor already added to SPP data.")
    else:
        logger.info("Adding Levenshtein-distance priming predictor to SPP data.")

        priming_word_columns = ["MatchedPrimeWord", "TargetWord"]
        matched_word_pairs = spp_data.dataframe[priming_word_columns].copy()

        # The priming OLD is the difference between related and matched unrelated pair OLDs
        matched_word_pairs[priming_levenshtein_column_name] = matched_word_pairs[priming_word_columns].apply(
            levenshtein_distance_local, axis=1) - spp_data.dataframe[levenshtein_column_name]

        spp_data.add_word_pair_keyed_predictor(matched_word_pairs, merge_on=priming_word_columns)


# Add prime–target Levenshtein distance
def levenshtein_distance_local(word_pair):
    word_1, word_2 = word_pair
    return levenshtein_distance(word_1, word_2)


def add_elexicon_predictor(spp_data: SppData,
                           elexicon_dataframe: pandas.DataFrame,
                           predictor_name: str,
                           prime_or_target: str):
    assert (prime_or_target in ["Prime", "Target"])

    # elex_prime_<predictor_name> or
    # elex_target_<predictor_name>
    new_predictor_name = f"elex_{prime_or_target.lower()}_" + predictor_name

    # PrimeWord or
    # TargetWord
    key_name = f"{prime_or_target}Word"

    # Don't bother training the model until we know we need it
    if spp_data.predictor_exists_with_name(new_predictor_name):
        logger.info(f"Elexicon predictor '{new_predictor_name}' already added to SPP data.")
    else:

        logger.info(f"Adding Elexicon predictor '{new_predictor_name} to SPP data.")

        # Dataframe with two columns: 'Word', [predictor_name]
        predictor = elexicon_dataframe[["Word", predictor_name]]

        # We'll join on PrimeWord first
        predictor = predictor.rename(columns={
            "Word": key_name,
            predictor_name: new_predictor_name
        })

        spp_data.add_word_keyed_predictor(predictor, key_name, new_predictor_name)


def add_predictors_for_model(model, spp_data: SppData):
    """
    Add all available predictors from this model.
    """

    for distance_type in DistanceType:

        if spp_data.predictor_exists_with_name(spp_data.predictor_name_for_model(model, distance_type)):
            logger.info(f"Predictor for '{model.name}' using '{distance_type.name}' already added to SPP data.")
        else:
            logger.info(f"Adding model predictor for '{model.name}' using '{distance_type.name}' to SPP data.")
            model.train()
            spp_data.add_model_predictor(model, distance_type)

        if spp_data.predictor_exists_with_name(spp_data.priming_predictor_name_for_model(model, distance_type)):
            logger.info(f"Priming predictor for '{model.name}' using '{distance_type.name}' already added to SPP data.")
        else:
            logger.info(f"Adding model priming predictor for '{model.name}' using '{distance_type.name}' to SPP data.")
            model.train()
            spp_data.add_model_priming_predictor(model, distance_type)


def save_wordlist(vocab: Set[str]):
    """
    Saves the vocab to a file
    """
    wordlist_path = os.path.join(Preferences.spp_results_dir, 'spp_wordlist.txt')
    separator = " "

    logger.info(f"Saving SPP word list to {wordlist_path}.")

    with open(wordlist_path, mode="w", encoding="utf-8") as wordlist_file:
        for word in sorted(vocab):
            wordlist_file.write(word + separator)
        # Terminate with a newline XD
        wordlist_file.write("\n")


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
