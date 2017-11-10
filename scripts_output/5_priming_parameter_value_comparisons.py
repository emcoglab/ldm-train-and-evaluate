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
import math
from collections import defaultdict

import pandas
import seaborn
from matplotlib import pyplot
from pandas import DataFrame

from ..core.utils.logging import log_message, date_format
from ..core.utils.maths import DistanceType
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


def model_name_without_distance(r):
    if not r['Model category'] == "Predict":
        return f"{r['Model type']} {r['Embedding size']:.0f} r={r['Window radius']} {r['Corpus']}"
    else:
        return f"{r['Model type']} r={r['Window radius']} {r['Corpus']}"


def model_name_without_radius(r):
    if not r['Model category'] == "Predict":
        return f"{r['Model type']} {r['Embedding size']:.0f} {r['Distance type']} {r['Corpus']}"
    else:
        return f"{r['Model type']} {r['Distance type']} {r['Corpus']}"


def model_name_without_embedding_size(r):
    return f"{r['Model type']} r={r['Window radius']} {r['Distance type']} {r['Corpus']}"


def predict_models_only(df: DataFrame) -> DataFrame:
    return df[df["Model category"] == "Predict"]


def main():

    regression_df = load_data()

    b_corr_cos_distributions(regression_df)

    compare_param_values(regression_df, param_name="Window radius", param_values=Preferences.window_radii,
                         model_name_func=model_name_without_radius)
    compare_param_values(regression_df, param_name="Embedding size", param_values=Preferences.predict_embedding_sizes,
                         model_name_func=model_name_without_embedding_size, row_filter=predict_models_only)
    compare_param_values(regression_df, param_name="Distance type", param_values=[d.name for d in DistanceType],
                         model_name_func=model_name_without_distance)


def compare_param_values(regression_df: DataFrame, param_name, param_values, model_name_func, row_filter=None):
    """
    Compares all model parameter values against all others for all DVs.
    Produces figures for the comparison.
    :param regression_df: Regression results
    :param param_name: The name of the parameter to take. Should be a column name of `results_df`
    :param param_values: The possible values the parameter can take
    :param model_name_func: function which takes a row of `results_df` and produces a name for the model.
                            Should produce a name which is the same for each `param_value` of `param_name`, and is
                            otherwise unique.
    :param row_filter: optional function with which to filter rows `results_df`
    :return:
    """

    figures_dir = os.path.join(figures_base_dir, "parameter comparisons")

    all_win_counts = []

    for dv in DV_NAMES:

        this_dv_df = regression_df[regression_df["Dependent variable"] == dv].copy()

        # Distributions of winning-radius-vs-next-best bfs
        bf_distributions = defaultdict(list)

        if row_filter is not None:
            this_dv_df = row_filter(this_dv_df)

        # Get appropriate model name
        this_dv_df["Model name"] = this_dv_df.apply(model_name_func, axis=1)

        for model_name in this_dv_df["Model name"].unique():
            all_embed_df = this_dv_df[this_dv_df["Model name"] == model_name]
            all_embed_df = all_embed_df.sort_values("b10_approx", ascending=False).reset_index(drop=True)

            # add bfs for winning radius to results
            bf_distributions[all_embed_df[param_name][0]].append(all_embed_df["B10 approx"][0]/all_embed_df["B10 approx"][1])

        win_count_this_dv = []

        for param_value in param_values:

            # Add to results

            win_count_this_dv.append(
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

        for result in win_count_this_dv:
            all_win_counts.append([dv, param_name] + result)

        all_win_counts_df = DataFrame(win_count_this_dv, columns=["Dependent variable", param_name, "Number of times winner"])

        seaborn.set_context(context="paper", font_scale=1)
        plot = seaborn.barplot(x=all_win_counts_df[param_name], y=all_win_counts_df["Number of times winner"])

        plot.set_xlabel(param_name)
        plot.set_title(f"Number of times each {param_name.lower()} is the best for {dv}")

        plot.figure.savefig(os.path.join(figures_dir, f"priming {param_name.lower()} bf dist {dv}.png"), dpi=300)

        pyplot.close(plot.figure)

        # Heatmap

        heatmap_df = all_win_counts_df.pivot(index="Dependent Variable", columns=param_name, values="Number of times winner")

        plot = seaborn.heatmap(heatmap_df, square=True)
        pyplot.yticks(rotation=0)

        plot.figure.savefig(os.path.join(figures_dir, f"priming {param_name.lower()} heatmap.png"), dpi=300)

        pyplot.close(plot.figure)

        # Save values to csv

        all_win_counts_df.to_csv(os.path.join(Preferences.summary_dir, f"priming {param_name.lower()} wins.csv"), index=False)


def b_corr_cos_distributions(regression_df: DataFrame):

    figures_dir = os.path.join(figures_base_dir, "bf histograms")
    seaborn.set(style="white", palette="muted", color_codes=True)

    for dv_name in DV_NAMES:
        distribution = []

        filtered_df: DataFrame = regression_df.copy()
        filtered_df = filtered_df[filtered_df["Dependent variable"] == dv_name]

        filtered_df["model_name"] = filtered_df.apply(model_name_without_distance, axis=1)

        for model_name in set(filtered_df["model_name"]):
            cos_df: DataFrame = filtered_df.copy()
            cos_df = cos_df[cos_df["model_name"] == model_name]
            cos_df = cos_df[cos_df["distance_type"] == "cosine"]

            corr_df: DataFrame = filtered_df.copy()
            corr_df = corr_df[corr_df["model_name"] == model_name]
            corr_df = corr_df[corr_df["distance_type"] == "correlation"]

            # barf
            bf_cos = list(cos_df["b10_approx"])[0]
            bf_corr = list(corr_df["b10_approx"])[0]

            bf_cos_cor = bf_cos / bf_corr

            distribution.append(math.log10(bf_cos_cor))

        seaborn.set_context(context="paper", font_scale=1)
        plot = seaborn.distplot(distribution, kde=False, color="b")

        xlims = plot.axes.get_xlim()
        plot.axes.set_xlim(
            -max(math.fabs(xlims[0]), math.fabs(xlims[1])),
            max(math.fabs(xlims[0]), math.fabs(xlims[1]))
        )

        plot.set_xlabel("log BF (cos, corr)")
        plot.set_title(f"Distribution of log BF (cos > corr) for {dv_name}")

        plot.figure.savefig(os.path.join(figures_dir, f"priming bf dist {dv_name}.png"), dpi=300)

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
