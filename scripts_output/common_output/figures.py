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
from typing import List

import seaborn
from matplotlib import pyplot
from pandas import DataFrame

from .constants import BF_THRESHOLD
from .dataframe import model_name_without_distance, model_name_without_corpus_or_distance_or_radius
from ...core.utils.maths import DistanceType


# Utility functions

def xticks_as_percentages(grid):
    try:
        xtick_labels = grid.axes[0][0].get_xticklabels()
    # TODO: Figure out what error this would be
    except:
        xtick_labels = grid.axes[0].get_xticklabels()
    grid.set_xticklabels(['{:3.0f}%'.format(float(label.get_text()) * 100) for label in xtick_labels])


def yticks_as_percentages(grid):
    try:
        ytick_labels = grid.axes[0][0].get_yticklabels()
    except:
        ytick_labels = grid.axes[0].get_yticklabels()
    grid.set_yticklabels(['{:3.0f}%'.format(float(label.get_text()) * 100) for label in ytick_labels])


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
                             row="Test name",
                             col_wrap=2,
                             size=5, aspect=1,
                             margin_titles=True,
                             xlim=(0, 1), ylim=(0, 1))

    grid.map(pyplot.scatter, "Cosine test statistic", "Correlation test statistic")

    for ax in grid.axes.flat:
        ax.plot((0, 1), (0, 1), c="r", ls="-")

    if ticks_as_percentages:
        xticks_as_percentages(grid)
        yticks_as_percentages(grid)

    pyplot.subplots_adjust(top=0.92)
    grid.fig.suptitle(f"{name_prefix}: correlation- & cosine-distance test statistics")

    grid.savefig(os.path.join(figures_dir, f"{name_prefix} cosine vs correlation distance.png"), dpi=300)
    pyplot.close(grid.fig)


def model_performance_bar_graphs(results: DataFrame,
                                 window_radius: int,
                                 key_column_name: str,
                                 test_statistic_name: str,
                                 name_prefix: str,
                                 figures_base_dir: str,
                                 distance_type: DistanceType,
                                 bayes_factor_decorations: bool=False,
                                 extra_h_line_at: float=None,
                                 ticks_as_percentages: bool=False):

    figures_dir = os.path.join(figures_base_dir, "model performance bar graphs")

    seaborn.set_style("ticks")

    filtered_df: DataFrame = results.copy()
    filtered_df = filtered_df[filtered_df["Window radius"] == window_radius]
    filtered_df = filtered_df[filtered_df["Distance type"] == distance_type.name]

    # Don't want to show PPMI (10000)
    # This only applies for synonym tests, but it doesn't cause a problem if it's not present
    filtered_df = filtered_df[filtered_df["Model type"] != "PPMI (10000)"]

    # Model name doesn't need to include corpus or distance, since those are fixed
    filtered_df["Model name"] = filtered_df.apply(model_name_without_corpus_or_distance_or_radius, axis=1)

    seaborn.set_context(context="paper", font_scale=1)
    grid = seaborn.FacetGrid(
        filtered_df,
        row=key_column_name, col="Corpus",
        margin_titles=True,
        size=2.5,
        ylim=(0, 1))

    grid.set_xticklabels(rotation=-90)

    if ticks_as_percentages:
        yticks_as_percentages(grid)

    # Plot the bars
    grid.map(seaborn.barplot, "Model name", test_statistic_name, order=[
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
    ])

    if extra_h_line_at is not None:
        # Plot the chance line
        grid.map(pyplot.axhline, y=extra_h_line_at, linestyle="solid", color="xkcd:bright red")

    if bayes_factor_decorations:
        grid.map(pyplot.axhline, y=1,              linestyle="solid",  color="xkcd:bright red")
        grid.map(pyplot.axhline, y=BF_THRESHOLD,   linestyle="dotted", color="xkcd:bright red")
        grid.map(pyplot.axhline, y=1/BF_THRESHOLD, linestyle="dotted", color="xkcd:bright red")
        grid.set(yscale="log")

    grid.set_ylabels(test_statistic_name)

    pyplot.subplots_adjust(top=0.92)
    grid.fig.suptitle(f"Model scores for radius {window_radius} using {distance_type.name} distance")

    figure_name = f"{name_prefix} r={window_radius} {distance_type.name} ({test_statistic_name}).png"

    grid.savefig(os.path.join(figures_dir, figure_name), dpi=300)

    pyplot.close(grid.fig)
