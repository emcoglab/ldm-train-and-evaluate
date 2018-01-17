"""
===========================
Creating and manipulating figures.
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
from collections import defaultdict
from typing import List

import seaborn
from matplotlib import pyplot
from pandas import DataFrame
import numpy

from .constants import BF_THRESHOLD
from .dataframe import model_name_without_distance, model_name_without_corpus_or_distance_or_radius, \
    predict_models_only
from ...core.utils.maths import DistanceType, CorrelationType
from ...preferences.preferences import Preferences

logger = logging.getLogger(__name__)


# Utility functions

def ticklabel_as_percentage(label):
    label_text = label.get_text()
    # Need to replace dashes with hyphens to help float conversion work properly
    label_text = label_text.replace("−", "-")
    # Some labels may be empty
    if not label_text:
        return ""
    else:
        return '{:3.0f}%'.format(float(label_text) * 100)


def xticks_as_percentages(grid):
    try:
        xtick_labels = grid.axes[0][0].get_xticklabels()
    except TypeError:
        xtick_labels = grid.axes[0].get_xticklabels()
    grid.set_xticklabels(list(map(ticklabel_as_percentage, xtick_labels)))


def yticks_as_percentages(grid):
    try:
        ytick_labels = grid.axes[0][0].get_yticklabels()
    except TypeError:
        ytick_labels = grid.axes[0].get_yticklabels()
    grid.set_yticklabels(list(map(ticklabel_as_percentage, ytick_labels)))


# Specific output graphs

def cosine_vs_correlation_scores(results: DataFrame,
                                 figures_base_dir: str,
                                 test_names: List[str],
                                 test_statistic_column_name: str,
                                 name_prefix: str,
                                 ticks_as_percentages=False):

    figures_dir = os.path.join(figures_base_dir, "effects of distance type")
    seaborn.set(style="white", palette="muted", color_codes=True)

    distribution = []
    for test_name in test_names:

        filtered_df: DataFrame = results.copy()
        filtered_df = filtered_df[filtered_df["Test name"] == test_name]

        filtered_df["Model name"] = filtered_df.apply(model_name_without_distance, axis=1)

        for model_name in set(filtered_df["Model name"]):
            cos_df: DataFrame = filtered_df.copy()
            cos_df = cos_df[cos_df["Model name"] == model_name]
            cos_df = cos_df[cos_df["Distance type"] == "cosine"]

            corr_df: DataFrame = filtered_df.copy()
            corr_df = corr_df[corr_df["Model name"] == model_name]
            corr_df = corr_df[corr_df["Distance type"] == "correlation"]

            # barf
            score_cos = list(cos_df[test_statistic_column_name])[0]
            score_corr = list(corr_df[test_statistic_column_name])[0]

            distribution.append([test_name, score_cos, score_corr])

    dist_df = DataFrame(distribution, columns=["Test name", "Cosine test statistic", "Correlation test statistic"])

    seaborn.set_context(context="paper", font_scale=1)

    grid = seaborn.FacetGrid(data=dist_df,
                             col="Test name",
                             col_wrap=2,
                             size=5, aspect=1,
                             margin_titles=True,
                             xlim=(-0.5, 1), ylim=(-0.5, 1))

    grid.map(pyplot.scatter, "Cosine test statistic", "Correlation test statistic")

    for ax in grid.axes.flat:
        ax.plot((-0.5, 1), (-0.5, 1), c="r", ls="-")

    if ticks_as_percentages:
        xticks_as_percentages(grid)
        yticks_as_percentages(grid)

    pyplot.subplots_adjust(top=0.92)
    grid.fig.suptitle(f"{name_prefix}: correlation- & cosine-distance test statistics")

    grid.savefig(os.path.join(figures_dir, f"{name_prefix} cosine vs correlation distance.png"), dpi=300)
    pyplot.close(grid.fig)


def pearson_vs_spearman_scores(results: DataFrame,
                               figures_base_dir: str,
                               test_names: List[str],
                               name_prefix: str):

    figures_dir = os.path.join(figures_base_dir, "effects of correlation type")
    seaborn.set(style="white", palette="muted", color_codes=True)

    distribution = []
    for test_name in test_names:
        for distance_type in DistanceType:

            filtered_df: DataFrame = results.copy()
            filtered_df = filtered_df[filtered_df["Test name"] == test_name]
            filtered_df = filtered_df[filtered_df["Distance type"] == distance_type.name]

            filtered_df["Model name"] = filtered_df.apply(model_name_without_distance, axis=1)

            for model_name in set(filtered_df["Model name"]):

                pearson_df: DataFrame = filtered_df.copy()
                pearson_df = pearson_df[pearson_df["Model name"] == model_name]
                pearson_df = pearson_df[pearson_df["Correlation type"] == CorrelationType.Pearson.name]

                spearman_df: DataFrame = filtered_df.copy()
                spearman_df = spearman_df[spearman_df["Model name"] == model_name]
                spearman_df = spearman_df[spearman_df["Correlation type"] == CorrelationType.Spearman.name]

                # barf
                score_pearson = list(pearson_df["Correlation"])[0]
                score_spearman = list(spearman_df["Correlation"])[0]

                distribution.append([test_name, score_pearson, score_spearman])

    dist_df = DataFrame(distribution, columns=["Test name", "Pearson", "Spearman"])

    seaborn.set_context(context="paper", font_scale=1)

    grid = seaborn.FacetGrid(data=dist_df,
                             col="Test name",
                             col_wrap=2,
                             size=5, aspect=1,
                             margin_titles=True,
                             xlim=(-0.5, 1), ylim=(-0.5, 1))

    grid.map(pyplot.scatter, "Pearson", "Spearman")

    for ax in grid.axes.flat:
        ax.plot((-0.5, 1), (-0.5, 1), c="r", ls="-")

    pyplot.subplots_adjust(top=0.92)
    grid.fig.suptitle(f"{name_prefix}: Pearson vs Spearman correlations")

    grid.savefig(os.path.join(figures_dir, f"{name_prefix} Pearson vs Spearman.png"), dpi=300)
    pyplot.close(grid.fig)


def model_performance_bar_graphs(results: DataFrame,
                                 window_radius: int,
                                 key_column_name: str,
                                 key_column_values: List,
                                 test_statistic_name: str,
                                 name_prefix: str,
                                 figures_base_dir: str,
                                 distance_type: DistanceType,
                                 bayes_factor_graph: bool=False,
                                 extra_h_line_at: float=None,
                                 ylim=None,
                                 ticks_as_percentages: bool=False):

    figures_dir = os.path.join(figures_base_dir, "model performance bar graphs")

    value_cap = 1e50

    seaborn.set_style("ticks")

    filtered_df: DataFrame = results.copy()
    filtered_df = filtered_df[filtered_df["Window radius"] == window_radius]
    filtered_df = filtered_df[filtered_df["Distance type"] == distance_type.name]
    filtered_df = filtered_df[filtered_df[key_column_name].isin(key_column_values)]

    # Don't want to show PPMI (10000)
    # This only applies for synonym tests, but it doesn't cause a problem if it's not present
    filtered_df = filtered_df[filtered_df["Model type"] != "PPMI (10000)"]

    # Model name doesn't need to include corpus or distance, since those are fixed
    filtered_df["Model name"] = filtered_df.apply(model_name_without_corpus_or_distance_or_radius, axis=1)

    # Apply value cap to Bayes factor graphs
    if bayes_factor_graph and (value_cap is not None):
        filtered_df[test_statistic_name] = filtered_df[test_statistic_name].apply(lambda x: min(x, value_cap))

    seaborn.set_context(context="paper", font_scale=0.5)

    grid = seaborn.FacetGrid(
        filtered_df,
        row=key_column_name, col="Corpus",
        margin_titles=True,
        size=2)

    grid.set_xticklabels(rotation=-90)

    if ticks_as_percentages:
        yticks_as_percentages(grid)

    # Plot the bars
    grid.map(
        seaborn.barplot, "Model name", test_statistic_name,
        order=[
            "log summed n-gram",
            "log n-gram",
        "Conditional probability",
        "Probability ratio",
        "PPMI",
        "Skip-gram 50",
        "Skip-gram 100",
        "Skip-gram 200",
        "Skip-gram 300",
        "Skip-gram 500",
        "CBOW 50",
        "CBOW 100",
        "CBOW 200",
        "CBOW 300",
        "CBOW 500",
        ]
    )

    if extra_h_line_at is not None:
        # Plot the chance line
        grid.map(pyplot.axhline, y=extra_h_line_at, linestyle="solid", color="xkcd:bright red")

    grid.set_ylabels(test_statistic_name)

    if ylim is not None:
        grid.set(ylim=ylim)

    if bayes_factor_graph:
        grid.map(pyplot.axhline, y=1,              linestyle="solid",  color="xkcd:bright red")
        grid.map(pyplot.axhline, y=BF_THRESHOLD,   linestyle="dotted", color="xkcd:bright red")
        grid.map(pyplot.axhline, y=1/BF_THRESHOLD, linestyle="dotted", color="xkcd:bright red")
        grid.set(yscale="log")

    grid.fig.tight_layout()

    pyplot.subplots_adjust(top=0.92)
    grid.fig.suptitle(f"Model scores for radius {window_radius} using {distance_type.name} distance")

    figure_name = f"{name_prefix} r={window_radius} {distance_type.name} ({test_statistic_name}).png"

    grid.savefig(os.path.join(figures_dir, figure_name), dpi=300)

    pyplot.close(grid.fig)


def score_vs_radius_line_graph(results: DataFrame,
                               key_column_name: str,
                               key_column_values: List,
                               test_statistic_name: str,
                               name_prefix: str,
                               figures_base_dir: str,
                               distance_type: DistanceType,
                               bayes_factor_decorations: bool=False,
                               additional_h_line_at: float=None,
                               ylim=None,
                               ticks_as_percenages: bool=False):

    figures_dir = os.path.join(figures_base_dir, "effects of radius")

    filtered_df: DataFrame = results.copy()
    filtered_df = filtered_df[filtered_df["Distance type"] == distance_type.name]
    filtered_df = filtered_df[filtered_df[key_column_name].isin(key_column_values)]

    # Don't need corpus, radius or distance, as they're fixed for each plot
    filtered_df["Model name"] = filtered_df.apply(model_name_without_corpus_or_distance_or_radius, axis=1)

    # We don't want this one
    # This only applies for synonym tests, but it doesn't cause a problem if it's not present
    filtered_df = filtered_df[filtered_df["Model name"] != "PPMI (10000)"]

    filtered_df = filtered_df.sort_values(by=["Model name", "Window radius"], ascending=True)
    filtered_df = filtered_df.reset_index(drop=True)

    seaborn.set_style("ticks")
    seaborn.set_context(context="paper", font_scale=1)
    grid = seaborn.FacetGrid(
        data=filtered_df,
        row=key_column_name, col="Corpus", hue="Model name",
        hue_order=[
            "log n-gram",
            "Conditional probability",
            "Probability ratio",
            "PPMI",
            "Skip-gram 50",
            "Skip-gram 100",
            "Skip-gram 200",
            "Skip-gram 300",
            "Skip-gram 500",
            "CBOW 50",
            "CBOW 100",
            "CBOW 200",
            "CBOW 300",
            "CBOW 500"
        ],
        palette=[
            "orange",
            "turquoise",
            "pink",
            "red",
            "#0000ff",
            "#2a2aff",
            "#5454ff",
            "#7e7eff",
            "#a8a8ff",
            "#00ff00",
            "#2aff2a",
            "#54ff54",
            "#7eff7e",
            "#a8ffa8",
        ],
        hue_kws=dict(
            marker=[
                "o",
                "o",
                "o",
                "o",
                "^",
                "^",
                "^",
                "^",
                "^",
                "^",
                "^",
                "^",
                "^",
                "^",
            ]
        ),
        margin_titles=True,
        legend_out=True,
        size=3.5)

    if ylim is not None:
        grid.set(ylim=ylim)

    if additional_h_line_at is not None:
        grid.map(pyplot.axhline, y=additional_h_line_at, ls=":", c=".5", label="", marker="")

    if bayes_factor_decorations:
        grid.map(pyplot.axhline, y=1,              linestyle="solid",  marker="", color="xkcd:bright red")
        grid.map(pyplot.axhline, y=BF_THRESHOLD,   linestyle="dotted", marker="", color="xkcd:bright red")
        grid.map(pyplot.axhline, y=1/BF_THRESHOLD, linestyle="dotted", marker="", color="xkcd:bright red")
        grid.set(yscale="log")

    grid.map(pyplot.plot, "Window radius", test_statistic_name)

    # Format yticks as percentages
    if ticks_as_percenages:
        yticks_as_percentages(grid)

    grid.add_legend(bbox_to_anchor=(1.15, 0.5))

    figure_name = f"{name_prefix} effect of radius {distance_type.name} ({test_statistic_name}).png"
    grid.savefig(os.path.join(figures_dir, figure_name), dpi=300)
    pyplot.close(grid.fig)


def score_vs_embedding_size_line_graph(results: DataFrame,
                                       name_prefix: str,
                                       window_radius: int,
                                       key_column_name: str,
                                       key_column_values: List[str],
                                       test_statistic_name: str,
                                       distance_type: DistanceType,
                                       figures_base_dir: str,
                                       additional_h_line_at: float=None,
                                       ylim=None,
                                       ticks_as_percentages: bool=False):

        figures_dir = os.path.join(figures_base_dir, "effects of embedding size")

        filtered_df: DataFrame = results.copy()
        filtered_df = filtered_df[filtered_df["Distance type"] == distance_type.name]
        filtered_df = filtered_df[filtered_df["Window radius"] == window_radius]
        filtered_df = filtered_df[filtered_df[key_column_name].isin(key_column_values)]

        # This graph doesn't make sense for count models
        filtered_df = predict_models_only(filtered_df)

        filtered_df = filtered_df.sort_values(by=["Corpus", "Model type", "Embedding size", key_column_name])
        filtered_df = filtered_df.reset_index(drop=True)

        seaborn.set_style("ticks")
        seaborn.set_context(context="paper", font_scale=1)

        grid = seaborn.FacetGrid(
            filtered_df,
            row=key_column_name, col="Corpus",
            hue="Model type",
            margin_titles=True,
            size=2,
            legend_out=True
        )

        if ylim is not None:
            grid.set(ylim=ylim)

        if additional_h_line_at is not None:
            grid.map(pyplot.axhline, y=additional_h_line_at, ls=":", c=".5", label="")

        grid.map(pyplot.plot, "Embedding size", test_statistic_name, marker="o")

        grid.set(xticks=Preferences.predict_embedding_sizes)

        if ticks_as_percentages:
            yticks_as_percentages(grid)

        grid.set_xlabels("Embedding size")
        grid.set_ylabels(test_statistic_name)

        grid.add_legend(title="Model", bbox_to_anchor=(1, 1))

        # Title
        title = f"Embedding size r={window_radius} ({distance_type.name})"
        pyplot.subplots_adjust(top=0.92)
        grid.fig.suptitle(title)

        figure_name = f"{name_prefix} Embedding size r={window_radius} {distance_type.name}.png"

        grid.savefig(os.path.join(figures_dir, figure_name), dpi=300)

        pyplot.close(grid.fig)


def compare_param_values_bf(test_results: DataFrame,
                            parameter_name: str,
                            bf_statistic_name: str,
                            figures_base_dir: str,
                            key_column_name: str,
                            key_column_values: List[str],
                            name_prefix: str,
                            parameter_values,
                            model_name_func,
                            row_filter=None):
    """
    Compares all model parameter values against all others for all tests using a Bayes factor ratio test.
    Produces figures for the comparison.
    :param test_results: Test results
    :param parameter_name: The name of the parameter to take. Should be a column name of `test_results`
    :param bf_statistic_name:
    :param figures_base_dir:
    :param key_column_name:
    :param key_column_values:
    :param name_prefix:
    :param parameter_values: The possible values the parameter can take
    :param model_name_func: function which takes a row of `test_results` and produces a name for the model.
                            Should produce a name which is the same for each `param_value` of `param_name`, and is
                            otherwise unique.
    :param row_filter: optional function with which to filter rows `test_results`
    :return:
    """

    figures_dir = os.path.join(figures_base_dir, "parameter comparisons")

    win_counts_all_tests = []
    win_fraction_all_tests = []

    # Consider each dependent variable separately
    for key_column_value in key_column_values:

        # Filter the regression results for this comparison
        regression_results_this_dv = test_results[test_results[key_column_name] == key_column_value].copy()
        # Apply further filters if necessary
        if row_filter is not None:
            regression_results_this_dv = row_filter(regression_results_this_dv)

        # Column containing the name of the models, not including information relating to the parameter being compared
        # (as this will be listed on another axis in any table or figure).
        regression_results_this_dv["Model name"] = regression_results_this_dv.apply(model_name_func, axis=1)

        # We will count "tied-best" parameter values, where "tied-best" means "indistinguishable from the best", rather
        # than "indistinguishable from next-best neighbour". This takes care of problematic examples arising from the
        # fact that the relation "is indistinguishable from" is not transitive.  Examples like this:
        #
        #   model rank	bf vs null	bf vs next	bf vs best
        #   ----------	----------	----------	----------
        #   1         	16        	2         	1
        #   2         	8         	2         	2
        #   3         	4         	2         	4
        #   4         	2         	2         	8
        #
        # (threshold ≈ 3)
        #
        # Only 1 and 2 would be "joint-best", even though no two neighbouring models are distinguishable.
        number_of_wins_for_param_value = defaultdict(int)

        # The maximum number of total wins is the number of total models
        n_models_overall = regression_results_this_dv.shape[0] / len(parameter_values)
        assert n_models_overall == int(n_models_overall)
        n_models_overall = int(n_models_overall)
        assert n_models_overall == regression_results_this_dv["Model name"].unique().shape[0]

        # Loop through models
        for model_name in regression_results_this_dv["Model name"].unique():
            # Collection of models differing only by the value of the parameter
            model_variations: DataFrame = regression_results_this_dv[regression_results_this_dv["Model name"] == model_name].copy()

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

                # Not all datasets have Bayesian information criterion values
                bic_available = "Model BIC" in model_variations.columns.values

                # BF for best model
                best_bayes_factor = model_variations[bf_statistic_name][0]
                best_param_value = model_variations[parameter_name][0]

                best_bic = model_variations["Model BIC"][0] if bic_available else None

                # If the bayes factor is sufficiently large, it may snap to numpy.inf.
                # If it's not, we can sensibly make a comparison.
                if not numpy.isinf(best_bayes_factor):
                    joint_best_models = model_variations[
                        # The actual best model
                        (model_variations[parameter_name] == best_param_value)
                        |
                        # Indistinguishable from best
                        (best_bayes_factor / model_variations[bf_statistic_name] < BF_THRESHOLD)
                    ]

                # If it is, we can fall back to comparing BICs (assuming they are not also numpy.inf)
                # because e and log are monotonic increasing on their domains
                elif bic_available:
                    log_bf_best_vs_competitor = (model_variations["Model BIC"] - best_bic) / 2
                    joint_best_models = model_variations[
                        # The actual best model
                        (model_variations[parameter_name] == best_param_value)
                        |
                        # Indistinguishable from best
                        (log_bf_best_vs_competitor < numpy.log(BF_THRESHOLD))
                    ]

                    if numpy.isinf(best_bic):
                        logger.warning("Encountered an infinite BIC")

                else:
                    logger.warning("Encountered an infinite Bayes factor and no BIC was available for fallback")
                    # We can only pick the ones which literally share a BF value with the best model
                    joint_best_models = model_variations[
                        model_variations[parameter_name] == best_param_value
                    ]

                # Record details of joint-best models
                for parameter_value in joint_best_models[parameter_name]:
                    number_of_wins_for_param_value[parameter_value] += 1

        # Add to all-DV win-counts
        for parameter_value in parameter_values:
            win_counts_all_tests.append([key_column_value, parameter_value, number_of_wins_for_param_value[parameter_value]])
            win_fraction_all_tests.append([key_column_value, parameter_value, number_of_wins_for_param_value[parameter_value] / n_models_overall])

    all_win_counts = DataFrame(win_counts_all_tests, columns=[key_column_name, parameter_name, "Number of times (joint-)best"])
    all_win_fractions = DataFrame(win_fraction_all_tests, columns=[key_column_name, parameter_name, "Fraction of times (joint-)best"])

    # Bar graph for all DVs
    seaborn.set_style("ticks")
    seaborn.set_context(context="paper", font_scale=1)
    grid = seaborn.FacetGrid(data=all_win_fractions, col=key_column_name, col_wrap=2, margin_titles=True, size=2.5, ylim=(0, 1))

    yticks_as_percentages(grid)

    grid.map(seaborn.barplot, parameter_name, "Fraction of times (joint-)best")

    grid.fig.savefig(os.path.join(figures_dir, f"{name_prefix} {parameter_name.lower()} all dvs.png"), dpi=300)
    pyplot.close(grid.fig)

    # Heatmap for all DVs

    heatmap_df = all_win_fractions.pivot(index=key_column_name, columns=parameter_name, values="Fraction of times (joint-)best")
    plot = seaborn.heatmap(heatmap_df,
                           square=True,
                           linewidths=0.5,
                           # cbar_kws={"shrink": .5},
                           cmap=seaborn.light_palette("green", as_cmap=True))
    pyplot.xticks(rotation=90)
    pyplot.yticks(rotation=0)

    plot.figure.set_size_inches(3, 2)
    pyplot.tight_layout()

    # Colorbar has % labels
    old_labels = plot.collections[0].colorbar.ax.get_yticklabels()
    plot.collections[0].colorbar.set_ticks([float(label.get_text()) for label in old_labels])
    plot.collections[0].colorbar.set_ticklabels(['{:3.0f}%'.format(float(label.get_text()) * 100) for label in old_labels])

    plot.figure.savefig(os.path.join(figures_dir, f"{name_prefix} {parameter_name.lower()} heatmap.png"), dpi=300)
    pyplot.close(plot.figure)


def model_data_scatter_plot(transcript: DataFrame,
                            name_prefix: str,
                            figures_dir: str):

    figures_dir = os.path.join(figures_dir, "model data fit")

    seaborn.set(style="white", palette="muted", color_codes=True)
    seaborn.set_context(context="paper", font_scale=1)

    grid = seaborn.lmplot(data=transcript,
                          col="Test name", col_wrap=2,
                          x="Model distance", y="Data similarity",
                          fit_reg=False,
                          sharex=False, sharey=False,
                          size=5, aspect=1,
                          scatter_kws={"s": 3})

    pyplot.subplots_adjust(top=0.92)
    grid.fig.suptitle(f"{name_prefix} model vs data")

    grid.savefig(os.path.join(figures_dir, f"{name_prefix} model vs data.png"), dpi=300)
    pyplot.close(grid.fig)
