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
from typing import Set, List, Callable, Optional

import statsmodels.formula.api as sm
from pandas import DataFrame, read_csv
from statsmodels.regression.linear_model import RegressionResults

from constants import DISTANCE_TYPES
from ..ldm.corpus.indexing import FreqDist
from ..ldm.evaluation.regression import RegressionResult, CalgaryData, variance_inflation_factors
from ..ldm.model.base import VectorSemanticModel, DistributionalSemanticModel
from ..ldm.model.count import LogCoOccurrenceCountModel, ConditionalProbabilityModel, ProbabilityRatioModel, PPMIModel
from ..ldm.model.ngram import LogNgramModel, PPMINgramModel, ProbabilityRatioNgramModel
from ..ldm.model.predict import SkipGramModel, CbowModel
from ..ldm.utils.logging import log_message, date_format
from ..ldm.utils.maths import DistanceType, levenshtein_distance
from ..ldm.preferences.preferences import Preferences

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

        freq_dist = FreqDist.load(corpus_metadata.freq_dist_path)

        for window_radius in Preferences.window_radii:

            # N-GRAM MODELS

            ngram_models = [
                LogNgramModel(corpus_metadata, window_radius, freq_dist),
                PPMINgramModel(corpus_metadata, window_radius, freq_dist),
                ProbabilityRatioNgramModel(corpus_metadata, window_radius, freq_dist)
            ]

            for model in ngram_models:
                for reference_word in calgary_data.reference_words:
                    calgary_data.add_model_predictor_fixed_reference(model, None, reference_word=reference_word, memory_map=True)
                # Skip min distance as if we're using associations rather than distances we would be looking at max
                # calgary_data.add_model_predictor_min_distance(model, None)
                calgary_data.add_model_predictor_reference_difference(model, None)
                model.untrain()

            del ngram_models

            # COUNT MODELS

            count_models = [
                # TODO: these model initialisers should be able to have FreqDists _optionally_ passed,
                # TODO: or else load them internally
                LogCoOccurrenceCountModel(corpus_metadata, window_radius, freq_dist),
                ConditionalProbabilityModel(corpus_metadata, window_radius, freq_dist),
                ProbabilityRatioModel(corpus_metadata, window_radius, freq_dist),
                PPMIModel(corpus_metadata, window_radius, freq_dist)
            ]

            for model in count_models:
                for distance_type in DISTANCE_TYPES:
                    for reference_word in calgary_data.reference_words:
                        calgary_data.add_model_predictor_fixed_reference(model, distance_type, reference_word=reference_word, memory_map=True)
                    calgary_data.add_model_predictor_min_distance(model, distance_type)
                    calgary_data.add_model_predictor_reference_difference(model, distance_type)
                model.untrain()

            del count_models

            # PREDICT MODELS

            for embedding_size in Preferences.predict_embedding_sizes:

                predict_models = [
                    SkipGramModel(corpus_metadata, window_radius, embedding_size),
                    CbowModel(corpus_metadata, window_radius, embedding_size)
                ]

                for model in predict_models:
                    for distance_type in DISTANCE_TYPES:
                        for reference_word in calgary_data.reference_words:
                            calgary_data.add_model_predictor_fixed_reference(model, distance_type, reference_word=reference_word, memory_map=True)
                        calgary_data.add_model_predictor_min_distance(model, distance_type)
                        calgary_data.add_model_predictor_reference_difference(model, distance_type)
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
                                             distance_type: Optional[DistanceType],
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

    baseline_regression_results: RegressionResults = sm.ols(
        formula=baseline_formula,
        data=regression_data).fit()
    model_regression_results: RegressionResults = sm.ols(
        formula=model_formula,
        data=regression_data).fit()

    model_design_matrix_df: DataFrame = DataFrame(data=model_regression_results.model.exog,
                                                  columns=model_regression_results.model.exog_names)

    vifs = variance_inflation_factors(model_design_matrix_df)
    vifs = vifs[vifs.index != 'Intercept']

    return RegressionResult(
        dv_name=f"{dv_name}_min_distance",
        model=model,
        distance_type=distance_type,
        baseline_r2=baseline_regression_results.rsquared,
        baseline_bic=baseline_regression_results.bic,
        model_r2=model_regression_results.rsquared,
        model_bic=model_regression_results.bic,
        model_t=model_regression_results.tvalues[model_predictor_name],
        model_p=model_regression_results.pvalues[model_predictor_name],
        model_beta=model_regression_results.params[model_predictor_name],
        df=model_regression_results.df_resid,
        max_vif=vifs.max(),
        max_vif_predictor=vifs.idxmax(),
    )


def run_single_model_regression_reference_difference(all_data: DataFrame,
                                                     distance_type: Optional[DistanceType],
                                                     dv_name: str,
                                                     model: DistributionalSemanticModel,
                                                     baseline_variable_names: List[str]):

    model_predictor_name = CalgaryData.predictor_name_for_model_reference_difference(model, distance_type)

    # drop rows with missing data in any relevant column, as this may vary from column to column
    regression_data = all_data[[dv_name] + baseline_variable_names + [model_predictor_name]].dropna(how="any")

    logger.info(f"Running {dv_name} distance-difference regressions for model {model_predictor_name}")

    # Formulae
    baseline_formula = f"{dv_name} ~ {' + '.join(baseline_variable_names)}"
    model_formula = f"{baseline_formula} + {model_predictor_name}"

    baseline_regression_results: RegressionResults = sm.ols(
        formula=baseline_formula,
        data=regression_data).fit()
    model_regression_results: RegressionResults = sm.ols(
        formula=model_formula,
        data=regression_data).fit()

    model_design_matrix_df: DataFrame = DataFrame(data=model_regression_results.model.exog,
                                                  columns=model_regression_results.model.exog_names)

    vifs = variance_inflation_factors(model_design_matrix_df)
    vifs = vifs[vifs.index != 'Intercept']

    return RegressionResult(
        dv_name=f"{dv_name}_diff_distance",
        model=model,
        distance_type=distance_type,
        baseline_r2=baseline_regression_results.rsquared,
        baseline_bic=baseline_regression_results.bic,
        model_r2=model_regression_results.rsquared,
        model_bic=model_regression_results.bic,
        model_t=model_regression_results.tvalues[model_predictor_name],
        model_p=model_regression_results.pvalues[model_predictor_name],
        model_beta=model_regression_results.params[model_predictor_name],
        df=model_regression_results.df_resid,
        max_vif=vifs.max(),
        max_vif_predictor=vifs.idxmax(),
    )


def run_dual_model_regression_both_references(all_data: DataFrame,
                                              distance_type: Optional[DistanceType],
                                              dv_name: str,
                                              model: DistributionalSemanticModel,
                                              reference_words: List[str],
                                              baseline_variable_names: List[str]):

    model_predictor_names = [CalgaryData.predictor_name_for_model_fixed_reference(model, distance_type, reference_word) for reference_word in reference_words]

    # drop rows with missing data in any relevant column, as this may vary from column to column
    regression_data = all_data[[dv_name] + baseline_variable_names + model_predictor_names].dropna(how="any")

    logger.info(f"Running {dv_name} dual-model regressions")

    # Formulae
    baseline_formula = f"{dv_name} ~ {' + '.join(baseline_variable_names)}"
    model_formula = f"{baseline_formula} + {' + '.join(model_predictor_names)}"

    baseline_regression_results: RegressionResults = sm.ols(
        formula=baseline_formula,
        data=regression_data).fit()
    model_regression_results: RegressionResults = sm.ols(
        formula=model_formula,
        data=regression_data).fit()

    model_design_matrix_df: DataFrame = DataFrame(data=model_regression_results.model.exog,
                                                  columns=model_regression_results.model.exog_names)

    vifs = variance_inflation_factors(model_design_matrix_df)
    vifs = vifs[vifs.index != 'Intercept']

    return RegressionResult(
        dv_name=f"{dv_name}_min_distance",
        model=model,
        distance_type=distance_type,
        baseline_r2=baseline_regression_results.rsquared,
        baseline_bic=baseline_regression_results.bic,
        model_r2=model_regression_results.rsquared,
        model_bic=model_regression_results.bic,
        # TODO: this is not a good way to deal with this
        model_t="; ".join([f"{predictor}: {model_regression_results.tvalues[predictor]}" for predictor in model_predictor_names]),
        model_p="; ".join([f"{predictor}: {model_regression_results.pvalues[predictor]}" for predictor in model_predictor_names]),
        model_beta="; ".join([f"{predictor}: {model_regression_results.params[predictor]}" for predictor in model_predictor_names]),
        df=model_regression_results.df_resid,
        max_vif=vifs.max(),
        max_vif_predictor=vifs.idxmax(),
    )


def run_single_model_regression_fixed_reference(all_data: DataFrame,
                                                distance_type: Optional[DistanceType],
                                                dv_name: str,
                                                model: DistributionalSemanticModel,
                                                reference_word: str,
                                                baseline_variable_names: List[str]):

    model_predictor_name = CalgaryData.predictor_name_for_model_fixed_reference(model, distance_type, reference_word)

    # drop rows with missing data in any relevant column, as this may vary from column to column
    regression_data = all_data[[dv_name] + baseline_variable_names + [model_predictor_name]].dropna(how="any")

    logger.info(f"Running {dv_name} fixed-distance regressions for model {model_predictor_name} and {reference_word}")

    # Formulae
    baseline_formula = f"{dv_name} ~ {' + '.join(baseline_variable_names)}"
    model_formula = f"{baseline_formula} + {model_predictor_name}"

    baseline_regression_results: RegressionResults = sm.ols(
        formula=baseline_formula,
        data=regression_data).fit()
    model_regression_results: RegressionResults = sm.ols(
        formula=model_formula,
        data=regression_data).fit()

    model_design_matrix_df: DataFrame = DataFrame(data=model_regression_results.model.exog,
                                                  columns=model_regression_results.model.exog_names)

    vifs = variance_inflation_factors(model_design_matrix_df)
    vifs = vifs[vifs.index != 'Intercept']

    return RegressionResult(
        dv_name=f"{dv_name}_{reference_word}_distance",
        model=model,
        distance_type=distance_type,
        baseline_r2=baseline_regression_results.rsquared,
        baseline_bic=baseline_regression_results.bic,
        model_r2=model_regression_results.rsquared,
        model_bic=model_regression_results.bic,
        model_t=model_regression_results.tvalues[model_predictor_name],
        model_p=model_regression_results.pvalues[model_predictor_name],
        model_beta=model_regression_results.params[model_predictor_name],
        df=model_regression_results.df_resid,
        max_vif=vifs.max(),
        max_vif_predictor=vifs.idxmax(),
    )


def run_all_model_regressions(all_data: DataFrame,
                              dependent_variable_names: List[str],
                              baseline_variable_names: List[str]):

    results: List[RegressionResult] = []

    for corpus_metadata in Preferences.source_corpus_metas:

        freq_dist = FreqDist.load(corpus_metadata.freq_dist_path)

        for window_radius in Preferences.window_radii:

            # N-GRAM MODELS

            ngram_models = [
                LogNgramModel(corpus_metadata, window_radius, freq_dist),
                PPMINgramModel(corpus_metadata, window_radius, freq_dist),
                ProbabilityRatioNgramModel(corpus_metadata, window_radius, freq_dist)
            ]

            for model in ngram_models:
                for dv_name in dependent_variable_names:
                    result = run_single_model_regression_fixed_reference(all_data, None, dv_name, model, "concrete", baseline_variable_names)
                    results.append(result)
                    result = run_single_model_regression_fixed_reference(all_data, None, dv_name, model, "abstract", baseline_variable_names)
                    results.append(result)
                    # result = run_single_model_regression_min_distance(all_data, None, dv_name, model, baseline_variable_names)
                    # results.append(result)
                    result = run_single_model_regression_reference_difference(all_data, None, dv_name, model, baseline_variable_names)
                    results.append(result)
                    # TODO: this shouldn't need to be hard-coded
                    result = run_dual_model_regression_both_references(all_data, None, dv_name, model, ["concrete", "abstract"], baseline_variable_names)
                    results.append(result)

                model.untrain()
            del ngram_models

            # Count models

            count_models = [
                LogCoOccurrenceCountModel(corpus_metadata, window_radius, freq_dist),
                ConditionalProbabilityModel(corpus_metadata, window_radius, freq_dist),
                ProbabilityRatioModel(corpus_metadata, window_radius, freq_dist),
                PPMIModel(corpus_metadata, window_radius, freq_dist)
            ]

            for model in count_models:
                for distance_type in DISTANCE_TYPES:
                    for dv_name in dependent_variable_names:
                        result = run_single_model_regression_fixed_reference(all_data, distance_type, dv_name, model, "concrete", baseline_variable_names)
                        results.append(result)
                        result = run_single_model_regression_fixed_reference(all_data, distance_type, dv_name, model, "abstract", baseline_variable_names)
                        results.append(result)
                        result = run_single_model_regression_min_distance(all_data, distance_type, dv_name, model, baseline_variable_names)
                        results.append(result)
                        result = run_single_model_regression_reference_difference(all_data, distance_type, dv_name, model, baseline_variable_names)
                        results.append(result)
                        result = run_dual_model_regression_both_references(all_data, distance_type, dv_name, model, ["concrete", "abstract"], baseline_variable_names)
                        results.append(result)

                model.untrain()
            del count_models

            # Predict models

            for embedding_size in Preferences.predict_embedding_sizes:

                predict_models = [
                    SkipGramModel(corpus_metadata, window_radius, embedding_size),
                    CbowModel(corpus_metadata, window_radius, embedding_size)
                ]

                for model in predict_models:
                    for distance_type in DISTANCE_TYPES:
                        for dv_name in dependent_variable_names:
                            result = run_single_model_regression_fixed_reference(all_data, distance_type, dv_name, model, "concrete", baseline_variable_names)
                            results.append(result)
                            result = run_single_model_regression_fixed_reference(all_data, distance_type, dv_name, model, "abstract", baseline_variable_names)
                            results.append(result)
                            result = run_single_model_regression_min_distance(all_data, distance_type, dv_name, model, baseline_variable_names)
                            results.append(result)
                            result = run_single_model_regression_reference_difference(all_data, distance_type, dv_name, model, baseline_variable_names)
                            results.append(result)
                            result = run_dual_model_regression_both_references(all_data, distance_type, dv_name, model, ["concrete", "abstract"], baseline_variable_names)
                            results.append(result)

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
