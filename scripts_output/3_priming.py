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
import math

import numpy
import seaborn
from matplotlib import pyplot
from pandas import DataFrame, isnull, read_csv

from .common_output.figures import model_performance_bar_graphs, score_vs_radius_line_graph, compare_param_values_bf
from .common_output.dataframe import add_model_category_column, model_name_without_radius, \
    model_name_without_embedding_size, predict_models_only, model_name_without_distance
from .common_output.tables import table_top_n_models
from ..core.utils.logging import log_message, date_format
from ..core.utils.maths import DistanceType
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)

DV_NAMES = [

    "LDT_200ms_Z",
    # "LDT_200ms_Acc",
    # "LDT_1200ms_Z",
    # "LDT_1200ms_Acc",

    "LDT_200ms_Z_Priming",
    # "LDT_200ms_Acc_Priming",
    # "LDT_1200ms_Z_Priming",
    # "LDT_1200ms_Acc_Priming",

    "NT_200ms_Z",
    # "NT_200ms_Acc",
    # "NT_1200ms_Z",
    # "NT_1200ms_Acc",

    "NT_200ms_Z_Priming",
    # "NT_200ms_Acc_Priming",
    # "NT_1200ms_Z_Priming",
    # "NT_1200ms_Acc_Priming"
]


figures_base_dir = os.path.join(Preferences.figures_dir, "priming")


def main():

    regression_results: DataFrame = load_data()

    # Get info about the dv, used for filtering
    graphs_df: DataFrame = regression_results.copy()
    graphs_df["dv_test_type"] = graphs_df.apply(lambda r: "LDT" if r["Dependent variable"].startswith("LDT") else "NT", axis=1)
    graphs_df["dv_measure"] = graphs_df.apply(lambda r: "Acc" if "Acc" in r["Dependent variable"]          else "Z-RT", axis=1)
    graphs_df["dv_soa"] = graphs_df.apply(lambda r: 200 if "_200ms" in r["Dependent variable"]       else 1200, axis=1)
    graphs_df["dv_priming"] = graphs_df.apply(lambda r: True if "Priming" in r["Dependent variable"]      else False, axis=1)
    for distance_type in DistanceType:

        # Group DVs
        for soa in [200, 1200]:
            for test_type in ["LDT", "NT"]:

                # Filter on this subset of DVs
                dv_names_this_set = [dv_name for dv_name in DV_NAMES if dv_name.startswith(test_type) and f"_{soa}ms" in dv_name]
                if not dv_names_this_set:
                    continue

                filtered_df: DataFrame = graphs_df.copy()
                filtered_df = filtered_df[filtered_df["Dependent variable"].isin(dv_names_this_set)]

                # Model performance bar graphs
                for radius in Preferences.window_radii:
                    logger.info(f"Making model performance bar graphs for {test_type} {soa}ms r={radius}")
                    model_performance_bar_graphs(
                        results=filtered_df,
                        window_radius=radius,
                        key_column_name="Dependent variable",
                        key_column_values=dv_names_this_set,
                        test_statistic_name="R-squared increase",
                        name_prefix=f"Priming ({test_type} {soa}ms)",
                        bayes_factor_graph=False,
                        distance_type=distance_type,
                        figures_base_dir=figures_base_dir,
                        # ylim=(0, None)
                    )
                    model_performance_bar_graphs(
                        results=filtered_df,
                        window_radius=radius,
                        key_column_name="Dependent variable",
                        key_column_values=dv_names_this_set,
                        test_statistic_name="B10 approx",
                        name_prefix=f"Priming ({test_type} {soa}ms)",
                        bayes_factor_graph=True,
                        distance_type=distance_type,
                        figures_base_dir=figures_base_dir
                    )

                # Score vs radius line graphs
                logger.info(f"Making score-v-radius graphs for {test_type} {soa}ms")
                score_vs_radius_line_graph(
                    results=filtered_df,
                    key_column_name="Dependent variable",
                    key_column_values=dv_names_this_set,
                    test_statistic_name="R-squared increase",
                    name_prefix=f"Priming ({test_type} {soa}ms)",
                    bayes_factor_decorations=False,
                    distance_type=distance_type,
                    figures_base_dir=figures_base_dir,
                    ylim=(0, None)
                )
                score_vs_radius_line_graph(
                    results=filtered_df,
                    key_column_name="Dependent variable",
                    key_column_values=dv_names_this_set,
                    test_statistic_name="B10 approx",
                    name_prefix=f"Priming ({test_type} {soa}ms)",
                    bayes_factor_decorations=True,
                    distance_type=distance_type,
                    figures_base_dir=figures_base_dir
                )

    for dv_name in DV_NAMES:
        for radius in Preferences.window_radii:
            for corpus_name in ["BNC", "BBC", "UKWAC"]:
                logger.info(f"Making model comparison matrix dv={dv_name}, r={radius}, c={corpus_name}")
                model_comparison_matrix(regression_results, dv_name, radius, corpus_name)

    # Summary tables
    logger.info("Making top-5 model tables overall")
    table_top_n_models(
        results=regression_results,
        top_n=5,
        key_column_values=DV_NAMES,
        sort_by_column="B10 approx",
        name_prefix="Priming",
        key_column_name="Dependent variable"
    )
    for distance_type in DistanceType:
        logger.info(f"Making top-5 model tables overall for {distance_type.name}")
        table_top_n_models(
            results=regression_results,
            top_n=5,
            key_column_values=DV_NAMES,
            sort_by_column="B10 approx",
            name_prefix="Priming",
            key_column_name="Dependent variable",
            distance_type=distance_type
        )

    b_corr_cos_distributions(regression_results)

    for dv_set in ["LDT", "NT"]:

        dv_names_this_set = [dv_name for dv_name in DV_NAMES if dv_set in dv_name]

        compare_param_values_bf(
            parameter_name="Window radius",
            test_results=regression_results,
            bf_statistic_name="B10 approx",
            key_column_name="Dependent variable",
            key_column_values=dv_names_this_set,
            figures_base_dir=figures_base_dir,
            name_prefix=f"Priming {dv_set}",
            parameter_values=Preferences.window_radii,
            model_name_func=model_name_without_radius
        )
        compare_param_values_bf(
            parameter_name="Embedding size",
            test_results=regression_results,
            bf_statistic_name="B10 approx",
            key_column_name="Dependent variable",
            key_column_values=dv_names_this_set,
            figures_base_dir=figures_base_dir,
            name_prefix=f"Priming {dv_set}",
            parameter_values=Preferences.predict_embedding_sizes,
            model_name_func=model_name_without_embedding_size,
            row_filter=predict_models_only
        )
        compare_param_values_bf(
            parameter_name="Distance type",
            test_results=regression_results,
            bf_statistic_name="B10 approx",
            key_column_name="Dependent variable",
            key_column_values=dv_names_this_set,
            figures_base_dir=figures_base_dir,
            name_prefix=f"Priming {dv_set}",
            parameter_values=[d.name for d in DistanceType],
            model_name_func=model_name_without_distance
        )


def b_corr_cos_distributions(regression_df: DataFrame):

    figures_dir = os.path.join(figures_base_dir, "bf histograms")
    seaborn.set(style="white", palette="muted", color_codes=True)

    for dv_name in DV_NAMES:
        distribution = []

        filtered_df: DataFrame = regression_df.copy()
        filtered_df = filtered_df[filtered_df["Dependent variable"] == dv_name]

        filtered_df["Model name"] = filtered_df.apply(model_name_without_distance, axis=1)

        for model_name in set(filtered_df["Model name"]):
            cos_df: DataFrame = filtered_df.copy()
            cos_df = cos_df[cos_df["Model name"] == model_name]
            if any(isnull(cos_df["Distance type"])):
                continue
            cos_df = cos_df[cos_df["Distance type"] == "cosine"]

            cor_df: DataFrame = filtered_df.copy()
            cor_df = cor_df[cor_df["Model name"] == model_name]
            if any(isnull(cor_df["Distance type"])):
                continue
            cor_df = cor_df[cor_df["Distance type"] == "correlation"]

            # barf
            bf_cos = list(cos_df["B10 approx"])[0]
            bf_corr = list(cor_df["B10 approx"])[0]

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


def model_comparison_matrix(spp_results_df: DataFrame, dv_name: str, radius: int, corpus_name: str):

    figures_dir = os.path.join(figures_base_dir, "heatmaps all models")

    seaborn.set(style="white")

    filtered_df: DataFrame = spp_results_df.copy()
    filtered_df = filtered_df[filtered_df["Dependent variable"] == dv_name]
    filtered_df = filtered_df[filtered_df["Window radius"] == radius]
    filtered_df = filtered_df[filtered_df["Corpus"] == corpus_name]

    filtered_df["Model name"] = filtered_df.apply(
        lambda r:
        f"{r['Distance type']} {r['Model type']} {r['Embedding size']:.0f}"
        if not numpy.math.isnan(r['Embedding size'])
        else f"{r['Distance type']} {r['Model type']}",
        axis=1
    )

    # filtered_df = filtered_df.sort_values("distance_type")

    # Make the model name the index so it will label the rows and columns of the matrix
    filtered_df = filtered_df.set_index('Model name')

    # filtered_df = filtered_df.sort_values(by=["Model type", "Embedding size", "Distance type"])

    # values - values[:, None] gives col-row
    # which is equivalent to row > col
    bf_matrix = filtered_df["Model BIC"].values - filtered_df["Model BIC"].values[:, None]

    bf_matrix = numpy.exp(bf_matrix)
    bf_matrix = numpy.log10(bf_matrix)
    n_rows, n_columns = bf_matrix.shape

    bf_matrix_df = DataFrame(bf_matrix, filtered_df.index, filtered_df.index)

    # Generate a mask for the upper triangle
    mask = numpy.zeros((n_rows, n_columns), dtype=numpy.bool)
    mask[numpy.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    seaborn.set_context(context="paper", font_scale=1)
    f, ax = pyplot.subplots(figsize=(16, 14))

    # Draw the heatmap with the mask and correct aspect ratio
    seaborn.heatmap(bf_matrix_df,
                    # mask=mask,
                    cmap=seaborn.diverging_palette(250, 15, s=75, l=50, center="dark", as_cmap=True),
                    square=True, cbar_kws={"shrink": .5})

    pyplot.xticks(rotation=-90)
    pyplot.yticks(rotation=0)

    figure_name = f"priming heatmap {dv_name} r={radius} {corpus_name}.png"
    figure_title = f"log(BF(row,col)) for {dv_name} ({corpus_name}, r={radius})"

    ax.set_title(figure_title)

    f.savefig(os.path.join(figures_dir, figure_name), dpi=300)

    pyplot.close(f)


def load_data() -> DataFrame:
    """
    Load a DataFrame from a collection of CSV fragments.
    """
    results_dir = Preferences.spp_results_dir
    separator = ","

    with open(os.path.join(results_dir, "regression.csv"), mode="r", encoding="utf-8") as regression_file:
        regression_df = read_csv(regression_file, sep=separator, header=0,
                                 converters={
                                     # Check if embedding size is the empty string,
                                     # as it would be for Count models
                                     "Embedding size": lambda v: int(v) if len(v) > 0 else numpy.nan
                                 })

    add_model_category_column(regression_df)

    return regression_df


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
