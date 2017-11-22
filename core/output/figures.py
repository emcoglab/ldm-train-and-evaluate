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

from ..output.dataframe import model_name_without_distance


# Utility functions

def xticks_as_percentages(grid):
    """
    Set
    """
    xtick_labels = grid.axes[0].get_xticklabels()
    grid.set_xticklabels(['{:3.0f}%'.format(float(label.get_text()) * 100) for label in xtick_labels])
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

    pyplot.subplots_adjust(top=0.92)
    grid.fig.suptitle(f"{name_prefix}: correlation- & cosine-distance test statistics")

    grid.savefig(os.path.join(figures_dir, f"{name_prefix} cosine vs correlation distance.png"), dpi=300)
    pyplot.close(grid.fig)
