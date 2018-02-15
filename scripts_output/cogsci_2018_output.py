"""
===========================
Figures for CogSci 2018 paper.
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

import sys
import logging
from os import path
from typing import List
from collections import defaultdict

import seaborn
import numpy
from pandas import DataFrame, read_csv
from matplotlib import pyplot

from .common_output.constants import BF_THRESHOLD
from .common_output.dataframe import add_model_category_column, model_name_without_radius, \
    model_name_without_embedding_size, model_name_without_distance
from ..core.utils.logging import log_message, date_format
from ..core.utils.maths import CorrelationType, DistanceType
from ..core.model.base import DistributionalSemanticModel
from ..core.evaluation.synonym import ToeflTest, EslTest, LbmMcqTest, SynonymResults
from ..core.evaluation.association import SimlexSimilarity, WordsimSimilarity, WordsimRelatedness, MenSimilarity, \
    ThematicRelatedness, AssociationResults
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)

OUTPUT_BASE_DIR = path.join(Preferences.evaluation_dir, "for publication", "cogsci2018")


def main():

    # Load results
    synonym_results = SynonymResults().load().data
    association_results = AssociationResults().load().data
    priming_results = load_priming_data()
    concreteness_results = load_calgary_data()

    # Add "Model category" column
    add_model_category_column(synonym_results)
    add_model_category_column(association_results)
    add_model_category_column(priming_results)
    add_model_category_column(concreteness_results)

    # Remove irrelevant results
    synonym_results = remove_truncated_results(synonym_results)
    association_results = remove_truncated_results(association_results)
    priming_results = remove_truncated_results(priming_results)
    concreteness_results = remove_truncated_results(concreteness_results)

    # Remove Spearman results
    association_results = association_results[association_results["Correlation type"] == CorrelationType.Pearson.name]

    # Export CSVs
    save_synonym_results_csv(synonym_results)
    save_similarity_results_csv(association_results)
    save_relatedness_results_csv(association_results)
    save_norms_results_csv(association_results)
    save_priming_results_csv(priming_results)
    save_concreteness_results_csv(concreteness_results)

    # Synonym tests

    synonym_test_names = [ToeflTest().name, EslTest().name, LbmMcqTest().name]
    for test_name in synonym_test_names:
        single_violin_plot(results=synonym_results[synonym_results["Test name"] == test_name],
                           test_statistic_name="Score",
                           test_name=test_name,
                           extra_h_line_at=0.25,
                           y_lim=(0, 1)
                           )
    synonym_heatmaps(synonym_results, synonym_test_names)

    # Association tests

    similarity_test_names = [SimlexSimilarity().name, WordsimSimilarity().name]
    relatedness_test_names = [WordsimRelatedness().name, MenSimilarity().name]
    norm_test_names = [ThematicRelatedness().name]

    for test_name in (similarity_test_names + relatedness_test_names + norm_test_names):
        # Different y-axis coregistrations
        if test_name in (similarity_test_names + relatedness_test_names):
            y_lim = (0, 0.85)
        elif test_name in norm_test_names:
            y_lim = (0, 0.30)
        else:
            raise ValueError()

        single_violin_plot(
            results=association_results[association_results["Test name"] == test_name],
            test_statistic_name="Correlation",
            test_name=test_name,
            y_lim=y_lim
        )
    association_heatmaps("Similarity", similarity_test_names, association_results)
    association_heatmaps("Relatedness", relatedness_test_names, association_results)
    association_heatmaps("Norms", norm_test_names, association_results)

    # Priming tests

    priming_ldt_dvs = ["LDT_200ms_Z", "LDT_200ms_Z_Priming"]
    priming_nt_dvs = ["NT_200ms_Z", "NT_200ms_Z_Priming"]
    for test_name in (priming_ldt_dvs + priming_nt_dvs):
        # Different y-axis coregistrations
        if test_name in ["LDT_200ms_Z", "NT_200ms_Z"]:
            y_lim = (0, 0.025)
        elif test_name in ["LDT_200ms_Z_Priming", "NT_200ms_Z_Priming"]:
            y_lim = (0, 0.055)
        else:
            raise ValueError()
        single_violin_plot(
            results=priming_results[priming_results["Dependent variable"] == test_name],
            test_statistic_name="R-squared increase",
            test_name=test_name,
            y_lim=y_lim
        )
    priming_heatmaps("LDT", priming_ldt_dvs, priming_results)
    priming_heatmaps("NT", priming_nt_dvs, priming_results)

    # Calgary tests

    calgary_test_names = [
        # "zRTclean_mean_diff_distance",
        "zRTclean_mean_dual_distance",
        # "Concrete_response_proportion_diff_distance",
        "Concrete_response_proportion_dual_distance",
    ]
    for test_name in calgary_test_names:
        single_violin_plot(
            results=concreteness_results[concreteness_results["Dependent variable"] == test_name],
            test_statistic_name="R-squared increase",
            test_name=test_name
        )
    calgary_heatmaps(calgary_test_names, concreteness_results)


def calgary_heatmaps(calgary_test_names, concreteness_results):
    single_param_heatmap(
        test_results=concreteness_results,
        parameter_name="Window radius",
        test_kind="Concreteness",
        test_column_name="Dependent variable",
        test_names=calgary_test_names,
        bf_statistic_name="log10 B10 approx",
        using_log10_bf=True,
        parameter_values=Preferences.window_radii,
        model_name_func=model_name_without_radius,
    )
    single_param_heatmap(
        test_results=concreteness_results[
            concreteness_results["Model category"] == DistributionalSemanticModel.MetaType.predict.name],
        parameter_name="Embedding size",
        test_kind="Concreteness",
        test_column_name="Dependent variable",
        test_names=calgary_test_names,
        bf_statistic_name="log10 B10 approx",
        using_log10_bf=True,
        parameter_values=Preferences.predict_embedding_sizes,
        model_name_func=model_name_without_embedding_size,
    )
    single_param_heatmap(
        test_results=concreteness_results[
            concreteness_results["Model category"] != DistributionalSemanticModel.MetaType.ngram.name],
        parameter_name="Distance type",
        test_kind="Concreteness",
        test_column_name="Dependent variable",
        test_names=calgary_test_names,
        bf_statistic_name="log10 B10 approx",
        using_log10_bf=True,
        parameter_values=[d.name for d in DistanceType],
        model_name_func=model_name_without_distance,
    )


def priming_heatmaps(test_kind, priming_dvs, priming_results):
    single_param_heatmap(
        test_results=priming_results,
        parameter_name="Window radius",
        test_kind=test_kind,
        test_column_name="Dependent variable",
        test_names=priming_dvs,
        bf_statistic_name="B10 approx", using_log10_bf=False,
        parameter_values=Preferences.window_radii,
        model_name_func=model_name_without_radius,
    )
    single_param_heatmap(
        test_results=priming_results[
            priming_results["Model category"] == DistributionalSemanticModel.MetaType.predict.name],
        parameter_name="Embedding size",
        test_kind=test_kind,
        test_column_name="Dependent variable",
        test_names=priming_dvs,
        bf_statistic_name="B10 approx", using_log10_bf=False,
        parameter_values=Preferences.predict_embedding_sizes,
        model_name_func=model_name_without_embedding_size,
    )
    single_param_heatmap(
        test_results=priming_results[
            priming_results["Model category"] != DistributionalSemanticModel.MetaType.ngram.name],
        parameter_name="Distance type",
        test_kind=test_kind,
        test_column_name="Dependent variable",
        test_names=priming_dvs,
        bf_statistic_name="B10 approx", using_log10_bf=False,
        parameter_values=[d.name for d in DistanceType],
        model_name_func=model_name_without_distance,
    )


def association_heatmaps(test_kind, association_test_names, association_results):
    single_param_heatmap(
        test_results=association_results,
        parameter_name="Window radius",
        test_kind=test_kind,
        test_column_name="Test name",
        test_names=association_test_names,
        bf_statistic_name="Log10 B10 approx", using_log10_bf=True,
        parameter_values=Preferences.window_radii,
        model_name_func=model_name_without_radius,
    )
    single_param_heatmap(
        test_results=association_results[
            association_results["Model category"] == DistributionalSemanticModel.MetaType.predict.name],
        parameter_name="Embedding size",
        test_kind=test_kind,
        test_column_name="Test name",
        test_names=association_test_names,
        bf_statistic_name="Log10 B10 approx", using_log10_bf=True,
        parameter_values=Preferences.predict_embedding_sizes,
        model_name_func=model_name_without_embedding_size,
    )
    single_param_heatmap(
        test_results=association_results[
            association_results["Model category"] != DistributionalSemanticModel.MetaType.ngram.name],
        parameter_name="Distance type",
        test_kind=test_kind,
        test_column_name="Test name",
        test_names=association_test_names,
        bf_statistic_name="Log10 B10 approx", using_log10_bf=True,
        parameter_values=[d.name for d in DistanceType],
        model_name_func=model_name_without_distance,
    )


def synonym_heatmaps(synonym_results, synonym_test_names):
    single_param_heatmap(
        test_results=synonym_results,
        parameter_name="Window radius",
        test_kind="Synonym",
        test_column_name="Test name",
        test_names=synonym_test_names,
        bf_statistic_name="B10",
        using_log10_bf=False,
        parameter_values=Preferences.window_radii,
        model_name_func=model_name_without_radius,
    )
    single_param_heatmap(
        test_results=synonym_results[
            synonym_results["Model category"] == DistributionalSemanticModel.MetaType.predict.name],
        parameter_name="Embedding size",
        test_kind="Synonym",
        test_column_name="Test name",
        test_names=synonym_test_names,
        bf_statistic_name="B10",
        using_log10_bf=False,
        parameter_values=Preferences.predict_embedding_sizes,
        model_name_func=model_name_without_embedding_size,
    )
    single_param_heatmap(
        test_results=synonym_results[
            synonym_results["Model category"] != DistributionalSemanticModel.MetaType.ngram.name],
        parameter_name="Distance type",
        test_kind="Synonym",
        test_column_name="Test name",
        test_names=synonym_test_names,
        bf_statistic_name="B10",
        using_log10_bf=False,
        parameter_values=[d.name for d in DistanceType],
        model_name_func=model_name_without_distance,
    )


def single_violin_plot(results: DataFrame,
                       test_statistic_name: str,
                       test_name: str,
                       extra_h_line_at: float = None,
                       y_lim=None):
    seaborn.set_style("whitegrid")

    local_results: DataFrame = results.copy()

    # !!!!
    # NOTE: For the purposes of display, we make all values positive! This should be acknowledged in any legend!
    # !!!!
    local_results[test_statistic_name] = local_results[test_statistic_name].apply(numpy.abs)

    seaborn.set_context(context="paper", font_scale=1)

    # Initialize the figure
    fig, ax = pyplot.subplots(figsize=(7, 6))

    if y_lim is not None:
        ax.set(ylim=y_lim)

    if extra_h_line_at is not None:
        pyplot.axhline(y=extra_h_line_at, linestyle="solid", color="xkcd:bright red")

    # pyplot.setp(ax.xaxis.get_majorticklabels(), rotation=-90)

    seaborn.violinplot(
        data=local_results,
        x="Model type", y=test_statistic_name,
        hue="Model category", hue_order=["N-gram", "Count", "Predict"], dodge=False,
        cut=0, inner=None, linewidth=0,
        scale="width",
        order=[
            # ngram
            DistributionalSemanticModel.ModelType.log_ngram.name,
            DistributionalSemanticModel.ModelType.probability_ratio_ngram.name,
            DistributionalSemanticModel.ModelType.ppmi_ngram.name,
            # count
            DistributionalSemanticModel.ModelType.log_cooccurrence.name,
            DistributionalSemanticModel.ModelType.conditional_probability.name,
            DistributionalSemanticModel.ModelType.probability_ratio.name,
            DistributionalSemanticModel.ModelType.ppmi.name,
            # predict
            DistributionalSemanticModel.ModelType.skip_gram.name,
            DistributionalSemanticModel.ModelType.cbow.name,
        ]
    )

    seaborn.swarmplot(
        data=local_results,
        x="Model type", y=test_statistic_name,
        marker="o", color="0", size=2,
        order=[
            # ngram
            DistributionalSemanticModel.ModelType.log_ngram.name,
            DistributionalSemanticModel.ModelType.probability_ratio_ngram.name,
            DistributionalSemanticModel.ModelType.ppmi_ngram.name,
            # count
            DistributionalSemanticModel.ModelType.log_cooccurrence.name,
            DistributionalSemanticModel.ModelType.conditional_probability.name,
            DistributionalSemanticModel.ModelType.probability_ratio.name,
            DistributionalSemanticModel.ModelType.ppmi.name,
            # predict
            DistributionalSemanticModel.ModelType.skip_gram.name,
            DistributionalSemanticModel.ModelType.cbow.name,
        ]
    )

    # Tweak the visual presentation
    ax.yaxis.grid(True)
    ax.set(xlabel="")
    ax.set(ylabel="")
    seaborn.despine(trim=True,
                    # top=True,
                    bottom=True,
                    left=True,
                    # right=True,
                    )
    ax.legend().set_visible(False)
    # fig.suptitle(f"{test_name}")
    fig.tight_layout()

    figure_name = f"violin plot {test_name} ({test_statistic_name}).png"
    pyplot.savefig(path.join(OUTPUT_BASE_DIR, figure_name), dpi=300)

    pyplot.close(fig)


def single_param_heatmap(test_results: DataFrame,
                         parameter_name: str,
                         parameter_values: List,
                         test_kind: str,
                         test_column_name: str,
                         test_names: List[str],
                         bf_statistic_name: str,
                         using_log10_bf: bool,
                         model_name_func,
                         ):

    win_counts_all_tests = []
    win_fraction_all_tests = []

    # Consider each test separately
    for test_name in test_names:

        local_test_results: DataFrame = test_results[test_results[test_column_name] == test_name].copy()

        # Column containing the name of the models, not including information relating to the parameter being compared
        # (as this will be listed on another axis in any table or figure).
        local_test_results["Model name"] = local_test_results.apply(model_name_func, axis=1)

        number_of_wins_for_param_value = defaultdict(int)

        # The maximum number of total wins is the number of total models
        n_models_overall = local_test_results.shape[0] / len(parameter_values)
        assert n_models_overall == int(n_models_overall)
        n_models_overall = int(n_models_overall)
        assert n_models_overall == local_test_results["Model name"].unique().shape[0]

        # Loop through models
        for model_name in local_test_results["Model name"].unique():
            # Collection of models differing only by the value of the parameter
            model_variations: DataFrame = local_test_results[
                local_test_results["Model name"] == model_name].copy()

            # Sort by BF(model, baseline)
            model_variations = model_variations.sort_values(bf_statistic_name, ascending=False).reset_index(drop=True)

            # Ignore any models which are indistinguishable from the baseline model
            n_remaining_models = model_variations[model_variations[bf_statistic_name] > BF_THRESHOLD].shape[0]

            # Some cases to consider

            # If no param values are distinguishable from baseline, there's nothing to remember
            if n_remaining_models == 0:
                continue

            # If there's just one best model: easy
            elif n_remaining_models == 1:
                # Record its details
                winning_parameter_value = model_variations[parameter_name][0]
                number_of_wins_for_param_value[winning_parameter_value] += 1

            # If there are multiple best models, we look at those which are indistinguishable from the best model
            elif n_remaining_models > 1:

                # BF for best model
                best_bf_statistic = model_variations[bf_statistic_name][0]
                best_param_value = model_variations[parameter_name][0]

                # If the bayes factor is sufficiently large, it may snap to numpy.inf.
                # If it's not, we can sensibly make a comparison.
                if not numpy.isinf(best_bf_statistic):
                    if not using_log10_bf:
                        joint_best_models = model_variations[
                            # The actual best model
                            (model_variations[parameter_name] == best_param_value)
                            |
                            # Indistinguishable from best
                            (best_bf_statistic / model_variations[bf_statistic_name] < BF_THRESHOLD)
                            ]
                    else:
                        joint_best_models = model_variations[
                            # The actual best model
                            (model_variations[parameter_name] == best_param_value)
                            |
                            # Indistinguishable from best
                            # Using logs so subtract instead of divide
                            (best_bf_statistic - model_variations[bf_statistic_name] < numpy.log10(BF_THRESHOLD))
                        ]
                else:
                    logger.warning("Encountered an apparently infinite Bayes factor")
                    # We can only pick the ones which literally share a BF value with the best model
                    joint_best_models = model_variations[
                        model_variations[parameter_name] == best_param_value
                    ]

                # Record details of joint-best models
                for parameter_value in joint_best_models[parameter_name]:
                    number_of_wins_for_param_value[parameter_value] += 1

        # Add to all-DV win-counts
        for parameter_value in parameter_values:
            win_counts_all_tests.append(
                [test_name, parameter_value, number_of_wins_for_param_value[parameter_value]])
            win_fraction_all_tests.append(
                [test_name, parameter_value, number_of_wins_for_param_value[parameter_value] / n_models_overall])

    # Heatmap for all DVs

    all_win_fractions = DataFrame(win_fraction_all_tests, columns=[test_column_name, parameter_name, "Fraction of times (joint-)best"])
    heatmap_df = all_win_fractions.pivot(index=test_column_name, columns=parameter_name, values="Fraction of times (joint-)best")
    heatmap_df = heatmap_df.reindex(index=test_names)

    plot = seaborn.heatmap(heatmap_df,
                           square=True,
                           linewidths=0.5,
                           cmap=seaborn.light_palette("green", as_cmap=True),
                           vmin=0, vmax=1)
    pyplot.xticks(rotation=90)
    pyplot.yticks(rotation=0)

    # plot.figure.set_size_inches(5, 2)
    pyplot.tight_layout()

    # Colorbar has % labels
    old_labels = plot.collections[0].colorbar.ax.get_yticklabels()
    plot.collections[0].colorbar.set_ticks([float(label.get_text()) for label in old_labels])
    plot.collections[0].colorbar.set_ticklabels(
        ['{:3.0f}%'.format(float(label.get_text()) * 100) for label in old_labels])

    plot.figure.savefig(path.join(OUTPUT_BASE_DIR, f"heatmap {parameter_name.lower()} {test_kind}.png"), dpi=300)
    pyplot.close(plot.figure)


def load_priming_data() -> DataFrame:
    """
    Load a DataFrame from a collection of CSV fragments.
    """
    results_dir = Preferences.spp_results_dir
    separator = ","

    with open(path.join(results_dir, "regression.csv"), mode="r", encoding="utf-8") as regression_file:
        regression_df = read_csv(regression_file, sep=separator, header=0,
                                 converters={
                                     # Check if embedding size is the empty string,
                                     # as it would be for Count models
                                     "Embedding size": lambda v: int(v) if len(v) > 0 else numpy.nan
                                 })
    return regression_df


def load_calgary_data() -> DataFrame:
    """
    Load a DataFrame from a collection of CSV fragments.
    """
    results_dir = Preferences.calgary_results_dir
    separator = ","

    with open(path.join(results_dir, "regression.csv"), mode="r", encoding="utf-8") as regression_file:
        regression_df = read_csv(regression_file, sep=separator, header=0,
                                 converters={
                                     # Check if embedding size is the empty string,
                                     # as it would be for Count models
                                     "Embedding size": lambda v: int(v) if len(v) > 0 else numpy.nan
                                 })
    return regression_df


def remove_truncated_results(results):
    results = results[results["Model type"] != "PPMI (10000)"]
    return results


def save_synonym_results_csv(synonym_results: DataFrame):
    columns = ["Test name", "Corpus", "Model type", "Model category", "Embedding size", "Window radius", "Distance type", "Score", "B10"]
    export_results_csv(
        synonym_results[columns].sort_values(by=columns),
        "results_synonym.csv"
    )


def save_similarity_results_csv(association_results: DataFrame):
    columns = ["Test name", "Corpus", "Model type", "Model category", "Embedding size", "Window radius", "Distance type", "Correlation", "B10 approx", "Log10 B10 approx"]
    export_results_csv(
        (
            association_results[
                association_results["Test name"].isin([SimlexSimilarity().name, WordsimSimilarity().name])]
        )[columns].sort_values(by=columns),
        "results_similarity.csv"
    )


def save_relatedness_results_csv(association_results: DataFrame):
    columns = ["Test name", "Corpus", "Model type", "Model category", "Embedding size", "Window radius", "Distance type", "Correlation", "B10 approx", "Log10 B10 approx"]
    export_results_csv(
        (
            association_results[
                association_results["Test name"].isin([WordsimRelatedness().name, MenSimilarity().name])]
        )[columns].sort_values(by=columns),
        "results_relatedness.csv"
    )


def save_norms_results_csv(association_results: DataFrame):
    columns = ["Test name", "Corpus", "Model type", "Model category", "Embedding size", "Window radius", "Distance type", "Correlation", "B10 approx", "Log10 B10 approx"]
    export_results_csv(
        (
            association_results[
                association_results["Test name"].isin([ThematicRelatedness().name])]
        )[columns].sort_values(by=columns),
        "results_norms.csv"
    )


def save_priming_results_csv(priming_results: DataFrame):
    columns = ["Dependent variable", "Corpus", "Model type", "Model category", "Embedding size", "Window radius", "Distance type", "Model R-squared", "R-squared increase", "B10 approx", "log10 B10 approx"]
    export_results_csv(
        (
            priming_results[priming_results["Dependent variable"].str.contains("Z")
                            & ~priming_results["Dependent variable"].str.contains("1200")]
        )[columns].sort_values(by=columns),
        "results_priming.csv"
    )


def save_concreteness_results_csv(concreteness_results: DataFrame):
    columns = ["Dependent variable", "Corpus", "Model type", "Model category", "Embedding size", "Window radius", "Distance type", "Model R-squared", "R-squared increase", "B10 approx", "log10 B10 approx"]
    export_results_csv(
        (
            concreteness_results[(concreteness_results["Dependent variable"] == "zRTclean_mean_dual_distance")
                                 | (concreteness_results["Dependent variable"] == "Concrete_response_proportion_dual_distance")]
        )[columns].sort_values(by=columns),
        "results_concreteness.csv"
    )


def export_results_csv(results: DataFrame, file_name: str):
    results.to_csv(path.join(OUTPUT_BASE_DIR, file_name), header=True, index=False)


if __name__ == '__main__':
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
