"""
===========================
Figures for comparing model parameter values with semantic priming tests.
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
from collections import defaultdict

import numpy
import pandas
import seaborn
from matplotlib import pyplot
from pandas import DataFrame

from ..core.utils.logging import log_message, date_format
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)

DV_NAMES = [

    "LDT_200ms_Z",
    "LDT_200ms_Acc",
    # "LDT_1200ms_Z",
    # "LDT_1200ms_Acc",

    "LDT_200ms_Z_Priming",
    "LDT_200ms_Acc_Priming",
    # "LDT_1200ms_Z_Priming",
    # "LDT_1200ms_Acc_Priming",

    "NT_200ms_Z",
    "NT_200ms_Acc",
    # "NT_1200ms_Z",
    # "NT_1200ms_Acc",

    "NT_200ms_Z_Priming",
    "NT_200ms_Acc_Priming",
    # "NT_1200ms_Z_Priming",
    # "NT_1200ms_Acc_Priming"
]

figures_base_dir = os.path.join(Preferences.figures_dir, "priming")


def main():

    results_df = load_data()

    compare_using_radius(results_df)
    compare_using_embedding_size(results_df)

    parameter_value_comparison(results_df)


def compare_using_radius(results_df: DataFrame):

    param_name = "Window radius"

    figures_dir = os.path.join(figures_base_dir, "parameter comparisons")

    results_all_dvs = []

    for dv in DV_NAMES:

        this_dv_df = results_df[results_df["Dependent variable"] == dv].copy()

        # Distributions of winning-radius-vs-next-best bfs
        bf_distributions = defaultdict(list)

        # model name, not including radius
        this_dv_df["Model name"] = this_dv_df.apply(
            lambda r:
            f"{r['Model type']} {r['Embedding size']:.0f} {r['Distance type']} {r['Corpus']}"
            if r['Embedding size'] is not None and not numpy.isnan(r['Embedding size'])
            else f"{r['Model type']} {r['Distance type']} {r['Corpus']}",
            axis=1
        )

        for model_name in this_dv_df["Model name"].unique():
            all_radii_df = this_dv_df[this_dv_df["Model name"] == model_name]
            all_radii_df = all_radii_df.sort_values("B10 approx", ascending=False).reset_index(drop=True)

            # add bfs for winning radius to results
            bf_distributions[all_radii_df[param_name][0]].append(
                all_radii_df["B10 approx"][0] / all_radii_df["B10 approx"][1])

        results_this_dv = []

        param_values = Preferences.window_radii

        for param_value in param_values:

            # Add to results

            results_this_dv.append(
                # value       number of wins
                [param_value, len(bf_distributions[param_value])]
            )

            # Make figure

            if len(bf_distributions[param_value]) > 1:
                seaborn.set_context(context="paper", font_scale=1)
                plot = seaborn.distplot(bf_distributions[param_value], kde=False, color="b")

                plot.axes.set_xlim(1, None)

                plot.set_xlabel("BF")
                plot.set_title(f"BFs for winning {param_name.lower()}={param_value} versus next competitor for {dv}")

                plot.figure.savefig(os.path.join(figures_dir, f"priming {param_name}={param_value} bf dist {dv}.png"),
                                    dpi=300)

                pyplot.close(plot.figure)

        for result in results_this_dv:
            results_all_dvs.append([dv, param_name] + result)

        results_this_dv_df = DataFrame(results_this_dv, columns=[param_name, "Number of times winner"])

        seaborn.set_context(context="paper", font_scale=1)
        plot = seaborn.barplot(x=results_this_dv_df[param_name], y=results_this_dv_df["Number of times winner"])

        plot.set_xlabel(param_name)
        plot.set_title(f"Number of times each {param_name.lower()} is the best for {dv}")

        plot.figure.savefig(os.path.join(figures_dir, f"priming {param_name.lower()} bf dist {dv}.png"), dpi=300)

        pyplot.close(plot.figure)


def compare_using_embedding_size(results_df: DataFrame):

    param_name = "Embedding size"

    figures_dir = os.path.join(figures_base_dir, "parameter comparisons")

    results_all_dvs = []

    for dv in DV_NAMES:

        this_dv_df = results_df[results_df["Dependent variable"] == dv].copy()

        # Distributions of winning-radius-vs-next-best bfs
        bf_distributions = defaultdict(list)

        # Only predict models have embedding sizes, and that's all we're interested in.
        this_dv_df = this_dv_df[this_dv_df["Model category"] == "Predict"]

        # model name, not including embedding size
        this_dv_df["Model name"] = this_dv_df.apply(
            lambda r:
            f"{r['Model type']} r={r['Window radius']} {r['Distance type']} {r['Corpus']}",
            axis=1
        )

        for model_name in this_dv_df["Model name"].unique():
            all_embed_df = this_dv_df[this_dv_df["Model name"] == model_name]
            all_embed_df = all_embed_df.sort_values("b10_approx", ascending=False).reset_index(drop=True)

            # add bfs for winning radius to results
            bf_distributions[all_embed_df[param_name][0]].append(all_embed_df["B10 approx"][0]/all_embed_df["B10 approx"][1])

        results_this_dv = []

        param_values = Preferences.predict_embedding_sizes

        for param_value in param_values:

            # Add to results

            results_this_dv.append(
                # value       number of wins
                [param_value, len(bf_distributions[param_value])]
            )

            # Make figure

            if len(bf_distributions[param_value]) > 1:

                seaborn.set_context(context="paper", font_scale=1)
                plot = seaborn.distplot(bf_distributions[param_value], kde=False, color="b")

                plot.axes.set_xlim(1, None)

                plot.set_xlabel("BF")
                plot.set_title(f"BFs for winning {param_name.lower()}={param_value} versus next competitor for {dv}")

                plot.figure.savefig(os.path.join(figures_dir, f"priming {param_name}={param_value} bf dist {dv}.png"), dpi=300)

                pyplot.close(plot.figure)

        for result in results_this_dv:
            results_all_dvs.append([dv, param_name] + result)

        results_this_dv_df = DataFrame(results_this_dv, columns=[param_name, "Number of times winner"])

        seaborn.set_context(context="paper", font_scale=1)
        plot = seaborn.barplot(x=results_this_dv_df[param_name], y=results_this_dv_df["Number of times winner"])

        plot.set_xlabel(param_name)
        plot.set_title(f"Number of times each {param_name.lower()} is the best for {dv}")

        plot.figure.savefig(os.path.join(figures_dir, f"priming {param_name.lower()} bf dist {dv}.png"), dpi=300)

        pyplot.close(plot.figure)


def parameter_value_comparison(results_df: DataFrame):

    all_results_df = DataFrame(results_all_dvs, columns=["DV name", "Parameter", "Value", "Number of times winner"])
    all_results_df.to_csv(os.path.join(summary_dir, "priming parameter wins.csv"), index=False)

    # Heatmaps

    radius_df = all_results_df[all_results_df["Parameter"] == "Radius"].copy()
    radius_df.drop("Parameter", axis=1, inplace=True)
    radius_df.rename(columns={"Value": "Radius"}, inplace=True)
    radius_df = radius_df.pivot(index="DV name", columns="Radius", values="Number of times winner")

    plot = seaborn.heatmap(radius_df, square=True)
    pyplot.yticks(rotation=0)

    figure_name = f"priming RADIUS heatmap.png"

    plot.figure.savefig(os.path.join(figures_dir, figure_name), dpi=300)

    pyplot.close(plot.figure)

    embedding_df = all_results_df[all_results_df["Parameter"] == "Embedding size"].copy()
    embedding_df.drop("Parameter", axis=1, inplace=True)
    embedding_df.rename(columns={"Value": "Embedding size"}, inplace=True)
    embedding_df = embedding_df.pivot(index="DV name", columns="Embedding size", values="Number of times winner")

    plot = seaborn.heatmap(embedding_df, square=True)
    pyplot.yticks(rotation=0)

    figure_name = f"priming EMBED heatmap.png"

    plot.figure.savefig(os.path.join(figures_dir, figure_name), dpi=300)

    pyplot.close(plot.figure)


def load_data() -> DataFrame:
    """
    Load a DataFrame from a collection of CSV fragments.
    """
    results_dir = Preferences.spp_results_dir
    separator = ","

    with open(os.path.join(results_dir, "regression.csv"), mode="r", encoding="utf-8") as regression_file:
        regression_df = pandas.read_csv(regression_file, sep=separator, header=0, dtype={"Embedding size": int})

    regression_df["Model category"] = regression_df.apply(lambda r: "Count" if pandas.isnull(regression_df["Embedding size"]) else "Predict", axis=1)

    return regression_df


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
