"""
===========================
Evaluate LDMs using semantic priming data.
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
from os import path
from typing import Set, List, Optional

from numpy import nan
import statsmodels.formula.api as sm
from pandas import DataFrame, isna, read_csv
from statsmodels.regression.linear_model import RegressionResults

from constants import DISTANCE_TYPES
from ldm.corpus.indexing import FreqDist
from ldm.evaluation.regression import SppData, RegressionResult, variance_inflation_factors
from ldm.model.base import DistributionalSemanticModel
from ldm.model.count import LogCoOccurrenceCountModel, ConditionalProbabilityModel, ProbabilityRatioModel, PPMIModel
from ldm.model.ngram import LogNgramModel, PPMINgramModel, ProbabilityRatioNgramModel
from ldm.model.predict import SkipGramModel, CbowModel
from ldm.utils.logging import log_message, date_format
from ldm.utils.maths import DistanceType, levenshtein_distance
from ldm.preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def main():
    spp_data = SppData()

    save_wordlist(spp_data.vocabulary)
    save_wordpairs(spp_data.word_pairs)

    add_elexicon_predictors(spp_data)
    add_all_model_predictors(spp_data)

    spp_data.export_csv_first_associate_only()

    regression_wrapper(spp_data)


def save_wordlist(vocab: Set[str]):
    """
    Saves the vocab to a file
    """
    wordlist_path = path.join(Preferences.spp_results_dir, 'wordlist.txt')
    separator = " "

    logger.info(f"Saving SPP word list to {wordlist_path}.")

    with open(wordlist_path, mode="w", encoding="utf-8") as wordlist_file:
        for word in sorted(vocab):
            wordlist_file.write(word + separator)
        # Terminate with a newline XD
        wordlist_file.write("\n")


def save_wordpairs(word_pairs):
    """
    Saves the used word pairs to a file.
    """
    wordpair_path = path.join(Preferences.spp_results_dir, "wordpairs.txt")
    # Separates each item in a pair
    item_separator = " : "
    # Separates each pair
    pair_separator = "\n"

    with open(wordpair_path, mode="w", encoding="utf-8") as wordpair_file:
        for word_pair in word_pairs:
            wordpair_file.write(item_separator.join(word_pair) + pair_separator)
        # Terminate file with a newline
        wordpair_file.write("\n")


def add_all_model_predictors(spp_data: SppData):
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
                spp_data.add_model_predictor(model, None, for_priming_effect=False, memory_map=True)
                spp_data.add_model_predictor(model, None, for_priming_effect=True, memory_map=True)
                model.untrain()

            del ngram_models

            # COUNT MODELS

            count_models = [
                LogCoOccurrenceCountModel(corpus_metadata, window_radius, freq_dist),
                ConditionalProbabilityModel(corpus_metadata, window_radius, freq_dist),
                ProbabilityRatioModel(corpus_metadata, window_radius, freq_dist),
                PPMIModel(corpus_metadata, window_radius, freq_dist)
            ]

            for model in count_models:
                for distance_type in DISTANCE_TYPES:
                    spp_data.add_model_predictor(model, distance_type, for_priming_effect=False, memory_map=True)
                    spp_data.add_model_predictor(model, distance_type, for_priming_effect=True, memory_map=True)
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
                        spp_data.add_model_predictor(model, distance_type, for_priming_effect=False, memory_map=True)
                        spp_data.add_model_predictor(model, distance_type, for_priming_effect=True, memory_map=True)
                    model.untrain()

                del predict_models


def regression_wrapper(spp_data: SppData):

    results_path = path.join(Preferences.spp_results_dir, "regression.csv")

    # Compute all models for non-priming data

    dependent_variable_names = [
        SppData.Columns.ldt_200_z,
        SppData.Columns.ldt_200_ac,
        SppData.Columns.ldt_1200_z,
        SppData.Columns.ldt_1200_ac,
        SppData.Columns.nt_200_z,
        SppData.Columns.nt_200_ac,
        SppData.Columns.nt_1200_z,
        SppData.Columns.nt_1200_ac,
    ]

    baseline_variable_names = [
        SppData.Columns.target_length,
        SppData.Columns.elex_target_log_wf,
        SppData.Columns.elex_target_old20,
        SppData.Columns.elex_target_pld20,
        SppData.Columns.elex_target_nsyll,
        SppData.Columns.prime_target_old,
    ]

    results = run_all_model_regressions(spp_data.dataframe, dependent_variable_names, baseline_variable_names,
                                        for_priming_effect=False)

    # Compute all models for priming data

    dependent_variable_priming_names = [
        SppData.Columns.ldt_200_z_priming,
        SppData.Columns.ldt_200_ac_priming,
        SppData.Columns.ldt_1200_z_priming,
        SppData.Columns.ldt_1200_ac_priming,
        SppData.Columns.nt_200_z_priming,
        SppData.Columns.nt_200_ac_priming,
        SppData.Columns.nt_1200_z_priming,
        SppData.Columns.nt_1200_ac_priming,
    ]

    baseline_variable_priming_names = [
        SppData.Columns.prime_target_old_priming
    ]

    priming_results = run_all_model_regressions(spp_data.dataframe, dependent_variable_priming_names, baseline_variable_priming_names,
                                                for_priming_effect=True)

    results.extend(priming_results)

    # Export results

    separator = ","
    with open(results_path, mode="w", encoding="utf-8") as results_file:
        # Print header
        results_file.write(separator.join(RegressionResult.headings()) + "\n")
        # Print results
        for result in results:
            results_file.write(separator.join(result.fields) + '\n')


def run_single_model_regression(all_data: DataFrame,
                                distance_type: Optional[DistanceType],
                                dv_name: str,
                                model: DistributionalSemanticModel,
                                baseline_variable_names: List[str],
                                for_priming_effect: bool):

    model_predictor_name = SppData.predictor_name_for_model(model, distance_type, for_priming_effect)

    # drop rows with missing data in any relevant column, as this may vary from column to column
    regression_data = all_data[[dv_name] + baseline_variable_names + [model_predictor_name]].dropna(how="any")

    logger.info(f"Running {dv_name} regressions for model {model_predictor_name}")

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
        dv_name=dv_name,
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
                              baseline_variable_names: List[str],
                              for_priming_effect: bool):

    results : List[RegressionResult] = []

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
                    result = run_single_model_regression(all_data, None, dv_name, model, baseline_variable_names, for_priming_effect)
                    results.append(result)

                # release memory
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
                        result = run_single_model_regression(all_data, distance_type, dv_name, model, baseline_variable_names, for_priming_effect)
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
                    for distance_type in DISTANCE_TYPES:
                        for dv_name in dependent_variable_names:
                            result = run_single_model_regression(all_data, distance_type, dv_name, model, baseline_variable_names, for_priming_effect)
                            results.append(result)

                    # release memory
                    model.untrain()
                del predict_models
    return results


def add_elexicon_predictors(spp_data: SppData):

    elexicon_dataframe: DataFrame = read_csv(Preferences.spp_elexicon_csv, header=0, encoding="utf-8")

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

    # Add prime-target Levenshtein distance

    def levenshtein_distance_local(word_pair):
        word_1, word_2 = word_pair
        if isna(word_1) or isna(word_2):
            return nan
        return levenshtein_distance(word_1, word_2)

    # Add Levenshtein distance column to data frame
    if spp_data.predictor_exists_with_name(SppData.Columns.prime_target_old):
        logger.info("Levenshtein-distance predictor already added to SPP data.")
    else:
        logger.info("Adding Levenshtein-distance predictor to SPP data.")

        word_pairs = spp_data.dataframe[[SppData.Columns.prime_word, SppData.Columns.target_word]].copy()
        word_pairs[SppData.Columns.prime_target_old] = word_pairs[[SppData.Columns.prime_word, SppData.Columns.target_word]].apply(levenshtein_distance_local, axis=1)

        spp_data.add_word_pair_keyed_predictor(word_pairs, merge_on=[SppData.Columns.prime_word, SppData.Columns.target_word])

    # Add Levenshtein priming distance column to data frame
    if spp_data.predictor_exists_with_name(SppData.Columns.prime_target_old_priming):
        logger.info("Levenshtein-distance priming predictor already added to SPP data.")
    else:
        logger.info("Adding Levenshtein-distance priming predictor to SPP data.")

        # need both of them when
        matched_pairs = spp_data.dataframe[[SppData.Columns.prime_word, SppData.Columns.target_word, SppData.Columns.matched_prime]].copy()

        # The priming OLD is the difference between related and matched unrelated pair OLDs
        matched_pairs[SppData.Columns.prime_target_old_priming] = matched_pairs[[SppData.Columns.matched_prime, SppData.Columns.target_word]].apply(
            levenshtein_distance_local, axis=1) - spp_data.dataframe[SppData.Columns.prime_target_old]

        spp_data.add_word_pair_keyed_predictor(matched_pairs[[SppData.Columns.prime_word, SppData.Columns.target_word, SppData.Columns.prime_target_old_priming]],
                                               merge_on=[SppData.Columns.prime_word, SppData.Columns.target_word])


def add_elexicon_predictor(spp_data: SppData,
                           elexicon_dataframe: DataFrame,
                           predictor_name: str,
                           prime_or_target: str):
    assert (prime_or_target in ["Prime", "Target"])

    # TODO: this logic shouldn't really be computed here, it should be in SppData.Columns
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
