"""
===========================
Figures for semantic priming tests.
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

    parameter_value_comparison(results_df)


def parameter_value_comparison(results_df: DataFrame):
    summary_dir = Preferences.summary_dir
    figures_dir = os.path.join(figures_base_dir, "parameter comparisons")

    results_all_dvs = []

    for dv_name in DV_NAMES:

        this_dv_df = results_df[results_df["dependent_variable"] == dv_name].copy()

        # RADIUS

        # Distributions of winning-radius-vs-next-best bfs
        radius_dist = defaultdict(list)

        # model name, not including radius
        this_dv_df["model_name"] = this_dv_df.apply(
            lambda r:
            f"{r['model_type']} {r['embedding_size']:.0f} {r['distance_type']} {r['corpus']}"
            if r['embedding_size'] is not None and not numpy.isnan(r['embedding_size'])
            else f"{r['model_type']} {r['distance_type']} {r['corpus']}",
            axis=1
        )

        for model_name in this_dv_df["model_name"].unique():
            all_radii_df = this_dv_df[this_dv_df["model_name"] == model_name]
            all_radii_df = all_radii_df.sort_values("b10_approx", ascending=False).reset_index(drop=True)

            # add bfs for winning radius to results
            radius_dist[all_radii_df["window_radius"][0]].append(all_radii_df["b10_approx"][0]/all_radii_df["b10_approx"][1])

        radius_results_this_dv = []

        for radius in Preferences.window_radii:

            # Add to results

            radius_results_this_dv.append(
                # value  number of wins
                [radius, len(radius_dist[radius])]
            )

            # Make figure

            if len(radius_dist[radius]) > 1:

                seaborn.set_context(context="paper", font_scale=1)
                plot = seaborn.distplot(radius_dist[radius], kde=False, color="b")

                plot.axes.set_xlim(1, None)

                plot.set_xlabel("BF")
                plot.set_title(f"BFs for winning RADIUS={radius} versus next competitor for {dv_name}")

                plot.figure.savefig(os.path.join(figures_dir, f"priming RADIUS={radius} bf dist {dv_name}.png"), dpi=300)

                pyplot.close(plot.figure)

        for result in radius_results_this_dv:
            results_all_dvs.append([dv_name, "radius"] + result)

        results_this_dv_df = DataFrame(radius_results_this_dv, columns=["Radius", "Number of times winner"])

        seaborn.set_context(context="paper", font_scale=1)
        plot = seaborn.barplot(x=results_this_dv_df["Radius"], y=results_this_dv_df["Number of times winner"])

        plot.set_xlabel("Radius")
        plot.set_title(f"Number of times each radius is the best for {dv_name}")

        plot.figure.savefig(os.path.join(figures_dir, f"priming RADIUS bf dist {dv_name}.png"), dpi=300)

        pyplot.close(plot.figure)

        # END RADIUS

        # EMBEDDING SIZE

        # Distributions of winning-radius-vs-next-best bfs
        embedding_size_dist = defaultdict(list)

        # model name, not including embedding size
        this_dv_df = results_df[results_df["dependent_variable"] == dv_name].copy()
        this_dv_df["model_name"] = this_dv_df.apply(
            lambda r:
            f"{r['model_type']} r={r['window_radius']} {r['distance_type']} {r['corpus']}",
            axis=1
        )

        # don't care about models without embedding sizes
        # TODO: store count/predict in regression results
        this_dv_df = this_dv_df[pandas.notnull(this_dv_df["embedding_size"])]

        # make embedding sizes ints
        # TODO: fix this at source
        this_dv_df["embedding_size"] = this_dv_df.apply(lambda r: int(r['embedding_size']), axis=1)

        for model_name in this_dv_df["model_name"].unique():
            all_embed_df = this_dv_df[this_dv_df["model_name"] == model_name]
            all_embed_df = all_embed_df.sort_values("b10_approx", ascending=False).reset_index(drop=True)

            # add bfs for winning radius to results
            embedding_size_dist[all_embed_df["embedding_size"][0]].append(all_embed_df["b10_approx"][0]/all_embed_df["b10_approx"][1])

        embedding_results_this_dv = []

        for embedding_size in Preferences.predict_embedding_sizes:

            # Add to results

            embedding_results_this_dv.append(
                # value          number of wins
                [embedding_size, len(embedding_size_dist[embedding_size])]
            )

            # Make figure

            if len(embedding_size_dist[embedding_size]) > 1:

                seaborn.set_context(context="paper", font_scale=1)
                plot = seaborn.distplot(embedding_size_dist[embedding_size], kde=False, color="b")

                plot.axes.set_xlim(1, None)

                plot.set_xlabel("BF")
                plot.set_title(f"BFs for winning EMBED={embedding_size} versus next competitor for {dv_name}")

                plot.figure.savefig(os.path.join(figures_dir, f"priming EMBED={embedding_size} bf dist {dv_name}.png"), dpi=300)

                pyplot.close(plot.figure)

        for result in embedding_results_this_dv:
            results_all_dvs.append([dv_name, "embedding size"] + result)

        results_this_dv_df = DataFrame(embedding_results_this_dv, columns=["Embedding size", "Number of times winner"])

        seaborn.set_context(context="paper", font_scale=1)
        plot = seaborn.barplot(x=results_this_dv_df["Embedding size"], y=results_this_dv_df["Number of times winner"])

        plot.set_xlabel("Embedding size")
        plot.set_title(f"Number of times each embedding size is the best for {dv_name}")

        plot.figure.savefig(os.path.join(figures_dir, f"priming EMBED bf dist {dv_name}.png"), dpi=300)

        pyplot.close(plot.figure)

        # END EMBEDDING SIZE

    all_results_df = DataFrame(results_all_dvs, columns=["DV name", "Parameter", "Value", "Number of times winner"])
    all_results_df.to_csv(os.path.join(summary_dir, "priming parameter wins.csv"), index=False)

    # Heatmaps

    radius_df = all_results_df[all_results_df["Parameter"] == "radius"].copy()
    radius_df.drop("Parameter", axis=1, inplace=True)
    radius_df.rename(columns={"Value": "Radius"}, inplace=True)
    radius_df = radius_df.pivot(index="DV name", columns="Radius", values="Number of times winner")

    plot = seaborn.heatmap(radius_df, square=True)
    pyplot.yticks(rotation=0)

    figure_name = f"priming RADIUS heatmap.png"

    plot.figure.savefig(os.path.join(figures_dir, figure_name), dpi=300)

    pyplot.close(plot.figure)

    embedding_df = all_results_df[all_results_df["Parameter"] == "embedding size"].copy()
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
        regression_df = pandas.read_csv(regression_file, sep=separator, header=0)

    return regression_df


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
