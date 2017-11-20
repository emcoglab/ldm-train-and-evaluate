"""
===========================
Evaluate using norms.
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
import math
import os
import sys
from collections import defaultdict

import numpy
import pandas
import seaborn
from matplotlib import pyplot
from pandas import DataFrame

from ..core.evaluation.association import AssociationResults, ColourEmotionAssociation, ThematicRelatedness
from ..core.utils.logging import log_message, date_format
from ..core.utils.maths import DistanceType
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)

TEST_NAMES = [ColourEmotionAssociation().name, ThematicRelatedness().name, ThematicRelatedness(only_use_response=1).name]

figures_base_dir = os.path.join(Preferences.figures_dir, "norms")

# The Bayes factor threshold at which we say one model is better than another
# This value from Jeffreys (1961) Theory of Probability.
BF_THRESHOLD = math.sqrt(10)


def model_name_without_distance(r):
    if r['Model category'] == "Predict":
        return f"{r['Model type']} {r['Embedding size']:.0f} r={r['Radius']} {r['Corpus']} ({r['Correlation type']})"
    else:
        return f"{r['Model type']} r={r['Radius']} {r['Corpus']} ({r['Correlation type']})"


def model_name_without_radius(r):
    if r['Model category'] == "Predict":
        return f"{r['Model type']} {r['Embedding size']:.0f} {r['Distance type']} {r['Corpus']} ({r['Correlation type']})"
    else:
        return f"{r['Model type']} {r['Distance type']} {r['Corpus']} ({r['Correlation type']})"


def model_name_without_embedding_size(r):
    return f"{r['Model type']} r={r['Radius']} {r['Distance type']} {r['Corpus']} ({r['Correlation type']})"


def predict_models_only(df: DataFrame) -> DataFrame:
    return df[df["Model category"] == "Predict"]


def main():

    association_df = AssociationResults().load().data

    association_df["Model category"] = association_df.apply(lambda r: "Count" if pandas.isnull(r["Embedding size"]) else "Predict", axis=1)

    compare_param_values(association_df, parameter_name="Radius", parameter_values=Preferences.window_radii,
                         model_name_func=model_name_without_radius)
    compare_param_values(association_df, parameter_name="Embedding size", parameter_values=Preferences.predict_embedding_sizes,
                         model_name_func=model_name_without_embedding_size, row_filter=predict_models_only)
    compare_param_values(association_df, parameter_name="Distance type", parameter_values=[d.name for d in DistanceType],
                         model_name_func=model_name_without_distance)


def compare_param_values(test_results: DataFrame, parameter_name, parameter_values, model_name_func, row_filter=None):
    """
    Compares all model parameter values against all others for all tests.
    Produces figures for the comparison.
    :param test_results: Test results
    :param parameter_name: The name of the parameter to take. Should be a column name of `test_results`
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
    for test_name in TEST_NAMES:

        # Filter the regression results for this comparison
        regression_results_this_dv = test_results[test_results["Test name"] == test_name].copy()
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
            model_variations = model_variations.sort_values("B10 approx", ascending=False).reset_index(drop=True)

            # Ignore any models which are indistinguishable from the baseline model
            n_remaining_models = model_variations[model_variations["B10 approx"] > BF_THRESHOLD].shape[0]

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
                best_bayes_factor = model_variations["B10 approx"][0]
                best_bic = model_variations["Model BIC"][0]
                best_param_value = model_variations[parameter_name][0]

                # If the bayes factor is sufficiently large, it may snap to numpy.inf.
                # If it's not, we can sensibly make a comparison.
                if not numpy.isinf(best_bayes_factor):
                    joint_best_models = model_variations[
                        # The actual best model
                        (model_variations[parameter_name] == best_param_value)
                        |
                        # Indistinguishable from best
                        (best_bayes_factor / model_variations["B10 approx"] < BF_THRESHOLD)

                    ]

                # If it is, we can fall back to comparing BICs (assuming they are not also numpy.inf)
                # because e and log are monotonic increasing on their domains
                else:
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

                # Record details of joint-best models
                for parameter_value in joint_best_models[parameter_name]:
                    number_of_wins_for_param_value[parameter_value] += 1

        # Add to all-DV win-counts
        for parameter_value in parameter_values:
            win_counts_all_tests.append([test_name, parameter_value, number_of_wins_for_param_value[parameter_value]])
            win_fraction_all_tests.append([test_name, parameter_value, number_of_wins_for_param_value[parameter_value] / n_models_overall])

    all_win_counts = DataFrame(win_counts_all_tests, columns=["Test name", parameter_name, "Number of times (joint-)best"])
    all_win_fractions = DataFrame(win_fraction_all_tests, columns=["Test name", parameter_name, "Fraction of times (joint-)best"])

    # Save values to csv
    all_win_counts.to_csv(os.path.join(Preferences.summary_dir, f"priming {parameter_name.lower()} win counts.csv"), index=False)
    all_win_fractions.to_csv(os.path.join(Preferences.summary_dir, f"priming {parameter_name.lower()} win fractions.csv"), index=False)

    # Bar graph for all DVs
    seaborn.set_style("ticks")
    seaborn.set_context(context="paper", font_scale=1)
    grid = seaborn.FacetGrid(data=all_win_fractions, col="Test name", col_wrap=2, margin_titles=True, size=2.5, ylim=(0, 1))
    # format y as percent
    grid.set_yticklabels(['{:3.0f}%'.format(float(label.get_text()) * 100) for label in grid.axes[0].get_yticklabels()])
    grid.map(seaborn.barplot, parameter_name, "Fraction of times (joint-)best")
    grid.fig.savefig(os.path.join(figures_dir, f"priming {parameter_name.lower()} all dvs.png"), dpi=300)
    pyplot.close(grid.fig)

    # Heatmap for all DVs
    heatmap_df = all_win_fractions.pivot(index="Test name", columns=parameter_name, values="Fraction of times (joint-)best")
    plot = seaborn.heatmap(heatmap_df, square=True, cmap=seaborn.light_palette("green", as_cmap=True))
    pyplot.xticks(rotation=90)
    pyplot.yticks(rotation=0)
    # Colorbar has % labels
    old_labels = plot.collections[0].colorbar.ax.get_yticklabels()
    plot.collections[0].colorbar.set_ticks([float(label.get_text()) for label in old_labels])
    plot.collections[0].colorbar.set_ticklabels(['{:3.0f}%'.format(float(label.get_text()) * 100) for label in old_labels])
    plot.figure.savefig(os.path.join(figures_dir, f"priming {parameter_name.lower()} heatmap.png"), dpi=300)
    pyplot.close(plot.figure)


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
