"""
===========================
Evaluate using concreteness norm data.
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
from typing import Set, List

import pandas
import statsmodels.formula.api as sm

from ..core.corpus.indexing import TokenIndexDictionary, FreqDist
from ..core.evaluation.regression import SppData, RegressionResult, CalgaryData
from ..core.model.base import VectorSemanticModel
from ..core.model.count import LogNgramModel, ConditionalProbabilityModel, ProbabilityRatioModel, PPMIModel
from ..core.model.predict import SkipGramModel, CbowModel
from ..core.utils.logging import log_message, date_format
from ..core.utils.maths import DistanceType, levenshtein_distance
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def main():
    calgary_data = CalgaryData()

    save_wordlist(calgary_data.vocabulary)

    add_all_model_predictors(calgary_data)

    add_elexicon_predictors(calgary_data)

    calgary_data.export_csv()

    regression_wrapper(calgary_data)


def save_wordlist(vocab: Set[str]):
    """
    Saves the vocab to a file
    """
    wordlist_path = os.path.join(Preferences.spp_results_dir, 'wordlist.txt')
    separator = " "

    logger.info(f"Saving SPP word list to {wordlist_path}.")

    with open(wordlist_path, mode="w", encoding="utf-8") as wordlist_file:
        for word in sorted(vocab):
            wordlist_file.write(word + separator)
        # Terminate with a newline XD
        wordlist_file.write("\n")


def add_all_model_predictors(calgary_data: CalgaryData):
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
                for distance_type in DistanceType:
                    for reference_word in calgary_data.reference_words:
                        calgary_data.add_model_predictor(model, distance_type, reference_word=reference_word, memory_map=True)
                model.untrain()

            del count_models

            # PREDICT MODELS

            for embedding_size in Preferences.predict_embedding_sizes:

                predict_models = [
                    SkipGramModel(corpus_metadata, window_radius, embedding_size),
                    CbowModel(corpus_metadata, window_radius, embedding_size)
                ]

                for model in predict_models:
                    for distance_type in DistanceType:
                        for reference_word in calgary_data.reference_words:
                            calgary_data.add_model_predictor(model, distance_type, reference_word=reference_word, memory_map=True)
                    model.untrain()

                del predict_models


def regression_wrapper(calgary_data: CalgaryData):

    results_path = os.path.join(Preferences.calgary_results_dir, "regression.csv")

    # Compute all models for non-priming data

    dependent_variable_names = [
        "zRTclean_mean",
        "ACC"
    ]

    baseline_variable_names = [
        "TargetLength",
        "elex_target_LgSUBTLWF",
        "elex_target_OLD",
        "elex_target_PLD",
        "elex_target_NSyll",
        "concrete_OrthLD",
        "abstract_OrthLD"
    ]

    results = run_all_model_regressions(calgary_data.dataframe, dependent_variable_names, baseline_variable_names)

    # Export results

    separator = ","
    with open(results_path, mode="w", encoding="utf-8") as results_file:
        # Print header
        results_file.write(separator.join(RegressionResult.headings()) + "\n")
        # Print results
        for result in results:
            results_file.write(separator.join(result.fields) + '\n')


def run_single_model_regression(all_data: pandas.DataFrame,
                                distance_type: DistanceType,
                                dv_name: str,
                                model: VectorSemanticModel,
                                baseline_variable_names: List[str]):

    model_predictor_name = CalgaryData.predictor_name_for_model(model, distance_type)

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

    return RegressionResult(
        dv_name,
        model,
        distance_type,
        baseline_regression.rsquared,
        baseline_regression.bic,
        model_regression.rsquared,
        model_regression.bic,
        model_regression.tvalues[model_predictor_name],
        model_regression.pvalues[model_predictor_name],
        model_regression.df_resid
    )


def run_all_model_regressions(all_data: pandas.DataFrame,
                              dependent_variable_names: List[str],
                              baseline_variable_names: List[str]):

    results : List[RegressionResult] = []

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
                        result = run_single_model_regression(all_data, distance_type, dv_name, model, baseline_variable_names)
                        results.append(result)

                # release memory
                model.untrain()
            del count_models

            # Predict models

            for embedding_size in Preferences.predict_embedding_sizes:

                predict_models = [
                    SkipGramModel(corpus_metadata, window_radius, embedding_size),
                    CbowModel(corpus_metadata, window_radius, embedding_size)
                ]

                for model in predict_models:
                    for distance_type in DistanceType:
                        for dv_name in dependent_variable_names:
                            result = run_single_model_regression(all_data, distance_type, dv_name, model, baseline_variable_names)
                            results.append(result)

                    # release memory
                    model.untrain()
                del predict_models
    return results


def add_elexicon_predictors(calgary_data: CalgaryData):

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
        add_elexicon_predictor(calgary_data, elexicon_dataframe, predictor_name)
        add_elexicon_predictor(calgary_data, elexicon_dataframe, predictor_name)

    # Add prime-target Levenshtein distance

    def levenshtein_distance_local(word_pair):
        word_1, word_2 = word_pair
        return levenshtein_distance(word_1, word_2)

    # Add Levenshtein distance column to data frame
    levenshtein_column_name = "OrthLD"
    if calgary_data.predictor_exists_with_name(levenshtein_column_name):
        logger.info("Levenshtein-distance predictor already added to Calgary data.")
    else:
        logger.info("Adding Levenshtein-distance predictor to Calgary data.")

        word_pairs = calgary_data.dataframe[["Word"]].copy()
        word_pairs[levenshtein_column_name] = word_pairs[["Word"]].apply(levenshtein_distance_local, axis=1)

        calgary_data.add_word_pair_keyed_predictor(word_pairs, merge_on=word_columns)

    # Add Levenshtein priming distance column to data frame
    priming_levenshtein_column_name = "PrimeTarget_OrthLD_Priming"
    if calgary_data.predictor_exists_with_name(priming_levenshtein_column_name):
        logger.info("Levenshtein-distance priming predictor already added to SPP data.")
    else:
        logger.info("Adding Levenshtein-distance priming predictor to SPP data.")

        priming_word_columns = ["MatchedPrimeWord", "TargetWord"]
        matched_word_pairs = calgary_data.dataframe[priming_word_columns].copy()

        # The priming OLD is the difference between related and matched unrelated pair OLDs
        matched_word_pairs[priming_levenshtein_column_name] = matched_word_pairs[priming_word_columns].apply(
            levenshtein_distance_local, axis=1) - calgary_data.dataframe[levenshtein_column_name]

        calgary_data.add_word_pair_keyed_predictor(matched_word_pairs, merge_on=priming_word_columns)


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


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
