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
from typing import Set, List, Callable

from pandas import DataFrame, read_csv
import statsmodels.formula.api as sm

from ..core.corpus.indexing import TokenIndexDictionary, FreqDist
from ..core.evaluation.regression import RegressionResult, CalgaryData
from ..core.model.base import VectorSemanticModel
from ..core.model.count import LogNgramModel, ConditionalProbabilityModel, ProbabilityRatioModel, PPMIModel, \
    LogSummedNgramModel
from ..core.model.predict import SkipGramModel, CbowModel
from ..core.utils.logging import log_message, date_format
from ..core.utils.maths import DistanceType, levenshtein_distance
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def main():
    calgary_data = CalgaryData()

    save_wordlist(calgary_data.vocabulary)

    add_lexical_predictors(calgary_data)

    add_all_model_predictors(calgary_data)

    regression_wrapper(calgary_data)


def save_wordlist(vocab: Set[str]):
    """
    Saves the vocab to a file
    """
    wordlist_path = os.path.join(Preferences.calgary_results_dir, 'wordlist.txt')
    separator = " "

    logger.info(f"Saving Calgary word list to {wordlist_path}.")

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
                # TODO: these model initialisers should be able to have TIDs and FDs _optionally_ passed, or load them internally
                LogSummedNgramModel(corpus_metadata, window_radius, token_index),
                LogNgramModel(corpus_metadata, window_radius, token_index),
                ConditionalProbabilityModel(corpus_metadata, window_radius, token_index, freq_dist),
                ProbabilityRatioModel(corpus_metadata, window_radius, token_index, freq_dist),
                PPMIModel(corpus_metadata, window_radius, token_index, freq_dist)
            ]

            for model in count_models:
                for distance_type in DistanceType:
                    for reference_word in calgary_data.reference_words:
                        calgary_data.add_model_predictor_fixed_distance(model, distance_type, reference_word=reference_word, memory_map=True)
                    calgary_data.add_model_predictor_min_distance(model, distance_type)
                    calgary_data.add_model_predictor_diff_distance(model, distance_type)
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
                            calgary_data.add_model_predictor_fixed_distance(model, distance_type, reference_word=reference_word, memory_map=True)
                        calgary_data.add_model_predictor_min_distance(model, distance_type)
                        calgary_data.add_model_predictor_diff_distance(model, distance_type)
                    model.untrain()

                del predict_models


def regression_wrapper(calgary_data: CalgaryData):

    results_path = os.path.join(Preferences.calgary_results_dir, "regression.csv")

    # Compute all models for non-priming data

    dependent_variable_names = [
        "zRTclean_mean",
        "Concrete_response_proportion",
        # "ACC"
    ]

    baseline_variable_names = [
        # Elexicon predictors:
        "elex_Length",
        "elex_LgSUBTLWF",
        "elex_OLD",
        "elex_PLD",
        "elex_NSyll",
        # Computed predictors:
        # "concrete_OrthLD",
        # "abstract_OrthLD",
        # "minimum_reference_OrthLD"
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


def run_single_model_regression_min_distance(all_data: DataFrame,
                                             distance_type: DistanceType,
                                             dv_name: str,
                                             model: VectorSemanticModel,
                                             baseline_variable_names: List[str]):

    model_predictor_name = CalgaryData.predictor_name_for_model_min_distance(model, distance_type)

    # drop rows with missing data in any relevant column, as this may vary from column to column
    regression_data = all_data[[dv_name] + baseline_variable_names + [model_predictor_name]].dropna(how="any")

    logger.info(f"Running {dv_name} minimum-distance regressions for model {model_predictor_name}")

    # Formulae
    baseline_formula = f"{dv_name} ~ {' + '.join(baseline_variable_names)}"
    model_formula = f"{baseline_formula} + {model_predictor_name}"

    baseline_regression = sm.ols(
        formula=baseline_formula,
        data=regression_data).fit()
    model_regression = sm.ols(
        formula=model_formula,
        data=regression_data).fit()

    return RegressionResult(
        f"{dv_name}_min_distance",
        model,
        distance_type,
        baseline_regression.rsquared,
        baseline_regression.bic,
        model_regression.rsquared,
        model_regression.bic,
        model_regression.tvalues[model_predictor_name],
        model_regression.pvalues[model_predictor_name],
        model_regression.params[model_predictor_name],
        model_regression.df_resid
    )


def run_single_model_regression_diff_distance(all_data: DataFrame,
                                              distance_type: DistanceType,
                                              dv_name: str,
                                              model: VectorSemanticModel,
                                              baseline_variable_names: List[str]):

    model_predictor_name = CalgaryData.predictor_name_for_model_diff_distance(model, distance_type)

    # drop rows with missing data in any relevant column, as this may vary from column to column
    regression_data = all_data[[dv_name] + baseline_variable_names + [model_predictor_name]].dropna(how="any")

    logger.info(f"Running {dv_name} distance-difference regressions for model {model_predictor_name}")

    # Formulae
    baseline_formula = f"{dv_name} ~ {' + '.join(baseline_variable_names)}"
    model_formula = f"{baseline_formula} + {model_predictor_name}"

    baseline_regression = sm.ols(
        formula=baseline_formula,
        data=regression_data).fit()
    model_regression = sm.ols(
        formula=model_formula,
        data=regression_data).fit()

    return RegressionResult(
        f"{dv_name}_diff_distance",
        model,
        distance_type,
        baseline_regression.rsquared,
        baseline_regression.bic,
        model_regression.rsquared,
        model_regression.bic,
        model_regression.tvalues[model_predictor_name],
        model_regression.pvalues[model_predictor_name],
        model_regression.params[model_predictor_name],
        model_regression.df_resid
    )


def run_dual_model_regression_both_distances(all_data: DataFrame,
                                             distance_type: DistanceType,
                                             dv_name: str,
                                             model: VectorSemanticModel,
                                             reference_words: List[str],
                                             baseline_variable_names: List[str]):

    model_predictor_names = [CalgaryData.predictor_name_for_model_fixed_distance(model, distance_type, reference_word) for reference_word in reference_words]

    # drop rows with missing data in any relevant column, as this may vary from column to column
    regression_data = all_data[[dv_name] + baseline_variable_names + model_predictor_names].dropna(how="any")

    logger.info(f"Running {dv_name} dual-model regressions")

    # Formulae
    baseline_formula = f"{dv_name} ~ {' + '.join(baseline_variable_names)}"
    model_formula = f"{baseline_formula} + {' + '.join(model_predictor_names)}"

    baseline_regression = sm.ols(
        formula=baseline_formula,
        data=regression_data).fit()
    model_regression = sm.ols(
        formula=model_formula,
        data=regression_data).fit()

    return RegressionResult(
        f"{dv_name}_dual_distance",
        model,
        distance_type,
        baseline_regression.rsquared,
        baseline_regression.bic,
        model_regression.rsquared,
        model_regression.bic,
        # TODO: this is not a good way to deal with this
        "; ".join([f"{predictor}: {model_regression.tvalues[predictor]}" for predictor in model_predictor_names]),
        "; ".join([f"{predictor}: {model_regression.pvalues[predictor]}" for predictor in model_predictor_names]),
        "; ".join([f"{predictor}: {model_regression.params[predictor]}" for predictor in model_predictor_names]),
        model_regression.df_resid
    )


def run_single_model_regression_fixed_distance(all_data: DataFrame,
                                               distance_type: DistanceType,
                                               dv_name: str,
                                               model: VectorSemanticModel,
                                               reference_word: str,
                                               baseline_variable_names: List[str]):

    model_predictor_name = CalgaryData.predictor_name_for_model_fixed_distance(model, distance_type, reference_word)

    # drop rows with missing data in any relevant column, as this may vary from column to column
    regression_data = all_data[[dv_name] + baseline_variable_names + [model_predictor_name]].dropna(how="any")

    logger.info(f"Running {dv_name} fixed-distance regressions for model {model_predictor_name} and {reference_word}")

    # Formulae
    baseline_formula = f"{dv_name} ~ {' + '.join(baseline_variable_names)}"
    model_formula = f"{baseline_formula} + {model_predictor_name}"

    baseline_regression = sm.ols(
        formula=baseline_formula,
        data=regression_data).fit()
    model_regression = sm.ols(
        formula=model_formula,
        data=regression_data).fit()

    return RegressionResult(
        f"{dv_name}_{reference_word}_distance",
        model,
        distance_type,
        baseline_regression.rsquared,
        baseline_regression.bic,
        model_regression.rsquared,
        model_regression.bic,
        model_regression.tvalues[model_predictor_name],
        model_regression.pvalues[model_predictor_name],
        model_regression.params[model_predictor_name],
        model_regression.df_resid
    )


def run_all_model_regressions(all_data: DataFrame,
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
                        result = run_single_model_regression_fixed_distance(all_data, distance_type, dv_name, model, "concrete", baseline_variable_names)
                        results.append(result)
                        result = run_single_model_regression_fixed_distance(all_data, distance_type, dv_name, model, "abstract", baseline_variable_names)
                        results.append(result)
                        result = run_single_model_regression_min_distance(all_data, distance_type, dv_name, model, baseline_variable_names)
                        results.append(result)
                        result = run_single_model_regression_diff_distance(all_data, distance_type, dv_name, model, baseline_variable_names)
                        results.append(result)
                        # TODO: this shouldn't need to be hard-coded
                        result = run_dual_model_regression_both_distances(all_data, distance_type, dv_name, model, ["concrete", "abstract"], baseline_variable_names)
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
                            result = run_single_model_regression_fixed_distance(all_data, distance_type, dv_name, model, "concrete", baseline_variable_names)
                            results.append(result)
                            result = run_single_model_regression_fixed_distance(all_data, distance_type, dv_name, model, "abstract", baseline_variable_names)
                            results.append(result)
                            result = run_single_model_regression_min_distance(all_data, distance_type, dv_name, model, baseline_variable_names)
                            results.append(result)
                            result = run_single_model_regression_diff_distance(all_data, distance_type, dv_name, model, baseline_variable_names)
                            results.append(result)
                            # TODO: this shouldn't need to be hard-coded
                            result = run_dual_model_regression_both_distances(all_data, distance_type, dv_name, model, ["concrete", "abstract"], baseline_variable_names)
                            results.append(result)

                    # release memory
                    model.untrain()
                del predict_models
    return results


def add_lexical_predictors(calgary_data: CalgaryData):

    elexicon_dataframe: DataFrame = read_csv(Preferences.calgary_elexicon_csv, header=0, encoding="utf-8")

    # Make sure the words are lowercase, as we'll use them as merging keys
    elexicon_dataframe['Word'] = elexicon_dataframe['Word'].str.lower()

    predictors_to_add = [
        "Length",
        "LgSUBTLWF",
        "OLD",
        "PLD",
        "NSyll"
    ]

    # Add Elexicon predictors

    for predictor_name in predictors_to_add:
        add_elexicon_predictor(calgary_data, elexicon_dataframe, predictor_name)

    # Add Levenshtein distances to reference words

    def levenshtein_distance_reference(ref_word: str) -> Callable:
        def levenshtein_distance_local(word: str) -> float:
            return levenshtein_distance(word, ref_word)
        return levenshtein_distance_local

    # Add Levenshtein distance columns to data frame

    ref_levenshtein_column_names = []
    for reference_word in calgary_data.reference_words:

        levenshtein_column_name = f"{reference_word}_OrthLD"
        ref_levenshtein_column_names.append(levenshtein_column_name)

        if calgary_data.predictor_exists_with_name(levenshtein_column_name):
            logger.info(f"{reference_word} Levenshtein-distance predictor already added to Calgary data.")
        else:
            logger.info(f"Adding {reference_word} Levenshtein-distance predictor to Calgary data.")

            ref_old_predictor = calgary_data.dataframe[["Word"]].copy()
            ref_old_predictor[levenshtein_column_name] = ref_old_predictor["Word"].apply(levenshtein_distance_reference(reference_word))

            calgary_data.add_word_keyed_predictor(ref_old_predictor, key_name="Word", predictor_name=levenshtein_column_name)

    # Add the minimum of the OLDs to the reference words

    min_old_column = calgary_data.dataframe[["Word"] + ref_levenshtein_column_names].copy()
    min_old_column["minimum_reference_OrthLD"] = min_old_column[ref_levenshtein_column_names].min(axis=1)
    # Take only column we want to actually add, plus the merge key
    min_old_column = min_old_column[["Word", "minimum_reference_OrthLD"]]

    calgary_data.add_word_keyed_predictor(min_old_column, key_name="Word", predictor_name="minimum_reference_OrthLD")


def add_elexicon_predictor(calgary_data: CalgaryData,
                           elexicon_dataframe: DataFrame,
                           predictor_name: str):

    # elex_<predictor_name>
    new_predictor_name = f"elex_" + predictor_name

    key_name = f"Word"

    # Don't bother training the model until we know we need it
    if calgary_data.predictor_exists_with_name(new_predictor_name):
        logger.info(f"Elexicon predictor '{new_predictor_name}' already added to Calgary data.")
    else:

        logger.info(f"Adding Elexicon predictor '{new_predictor_name} to Calgary data.")

        predictor = elexicon_dataframe[["Word", predictor_name]]

        predictor = predictor.rename(columns={predictor_name: new_predictor_name})

        calgary_data.add_word_keyed_predictor(predictor, key_name, new_predictor_name)


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
