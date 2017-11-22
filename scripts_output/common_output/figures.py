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
from ...preferences.preferences import Preferences


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


def score_vs_radius_line_graph(results: DataFrame,
                               key_column_name: str,
                               test_statistic_name: str,
                               name_prefix: str,
                               figures_base_dir: str,
                               distance_type: DistanceType,
                               bayes_factor_decorations: bool=False,
                               ticks_as_percenages: bool=False):

    figures_dir = os.path.join(figures_base_dir, "effects of radius")

    filtered_df: DataFrame = results.copy()
    filtered_df = filtered_df[filtered_df["Distance type"] == distance_type]

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
        size=3.5,
        ylim=(0, 1))
    grid.map(pyplot.plot, "Radius", test_statistic_name)

    # Format yticks as percentages
    if ticks_as_percenages:
        yticks_as_percentages(grid)

    if bayes_factor_decorations:
        grid.map(pyplot.axhline, y=1,              linestyle="solid",  color="xkcd:bright red")
        grid.map(pyplot.axhline, y=BF_THRESHOLD,   linestyle="dotted", color="xkcd:bright red")
        grid.map(pyplot.axhline, y=1/BF_THRESHOLD, linestyle="dotted", color="xkcd:bright red")
        grid.set(yscale="log")

    grid.add_legend(bbox_to_anchor=(1.15, 0.5))

    figure_name = f"{name_prefix} effect of radius {distance_type}.png"
    grid.savefig(os.path.join(figures_dir, figure_name), dpi=300)
    pyplot.close(grid.fig)


def figures_embedding_size(results: DataFrame,
                           name_prefix: str,
                           key_column_name: str,
                           key_column_value: str,
                           test_statistic_name: str,
                           distance_type: DistanceType,
                           figures_base_dir: str,
                           additional_h_line_at: float=None,
                           ticks_as_percentages: bool=False):

        figures_dir = os.path.join(figures_base_dir, "effects of embedding size")

        filtered_df: DataFrame = results.copy()
        filtered_df = filtered_df[filtered_df["Distance type"] == distance_type.name]
        filtered_df = filtered_df[filtered_df[key_column_name] == key_column_value]

        # This graph doesn't make sense for count models
        filtered_df = filtered_df[filtered_df["Model category"] == "Predict"]

        filtered_df = filtered_df.sort_values(by=["Corpus", "Model type", "Embedding size", "Radius"])
        filtered_df = filtered_df.reset_index(drop=True)

        seaborn.set_style("ticks")
        seaborn.set_context(context="paper", font_scale=1)
        grid = seaborn.FacetGrid(
            filtered_df,
            row="Radius", col="Corpus",
            margin_titles=True,
            size=2,
            ylim=(0, 1),
            legend_out=True
        )

        grid.map(pyplot.plot, "Embedding size", test_statistic_name, marker="o")

        if additional_h_line_at is not None:
            grid.map(pyplot.axhline, y=additional_h_line_at, ls=":", c=".5", label="")

        grid.set(xticks=Preferences.predict_embedding_sizes)

        if ticks_as_percentages:
            yticks_as_percentages(grid)

        grid.set_xlabels("Embedding size")
        grid.set_ylabels(test_statistic_name)

        grid.add_legend(title="Model", bbox_to_anchor=(1, 1))

        # Title
        title = f"{key_column_value} ({distance_type.name})"
        pyplot.subplots_adjust(top=0.92)
        grid.fig.suptitle(title)

        figure_name = f"{name_prefix} Embedding size {key_column_value} {distance_type.name}.png"

        grid.savefig(os.path.join(figures_dir, figure_name), dpi=300)

        pyplot.close(grid.fig)
