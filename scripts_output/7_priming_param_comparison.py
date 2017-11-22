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
import math
import os
import sys

import numpy
import pandas
import seaborn
from matplotlib import pyplot
from pandas import DataFrame

from .common_output.dataframe import model_name_without_radius, model_name_without_embedding_size, \
    model_name_without_distance, predict_models_only
from .common_output.figures import compare_param_values_bf
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


def main():

    regression_df = load_data()

    b_corr_cos_distributions(regression_df)

    compare_param_values_bf(
        parameter_name="Window radius",
        test_results=regression_df,
        bf_statistic_name="B10 approx",
        key_column_name="Dependent variable",
        key_column_values=DV_NAMES,
        figures_base_dir=figures_base_dir,
        name_prefix="Priming",
        parameter_values=Preferences.window_radii,
        model_name_func=model_name_without_radius
    )
    compare_param_values_bf(
        parameter_name="Embedding size",
        test_results=regression_df,
        bf_statistic_name="B10 approx",
        key_column_name="Dependent variable",
        key_column_values=DV_NAMES,
        figures_base_dir=figures_base_dir,
        name_prefix="Priming",
        parameter_values=Preferences.predict_embedding_sizes,
        model_name_func=model_name_without_embedding_size,
        row_filter=predict_models_only
    )
    compare_param_values_bf(
        parameter_name="Distance type",
        test_results=regression_df,
        bf_statistic_name="B10 approx",
        key_column_name="Dependent variable",
        key_column_values=DV_NAMES,
        figures_base_dir=figures_base_dir,
        name_prefix="Priming",
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
            cos_df = cos_df[cos_df["Distance type"] == "cosine"]

            corr_df: DataFrame = filtered_df.copy()
            corr_df = corr_df[corr_df["Model name"] == model_name]
            corr_df = corr_df[corr_df["Distance type"] == "correlation"]

            # barf
            bf_cos = list(cos_df["B10 approx"])[0]
            bf_corr = list(corr_df["B10 approx"])[0]

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
        regression_df = pandas.read_csv(regression_file, sep=separator, header=0,
                                        converters={
                                            # Check if embedding size is the empty string,
                                            # as it would be for Count models
                                            "Embedding size": lambda v: int(v) if len(v) > 0 else numpy.nan
                                        })

    regression_df["Model category"] = regression_df.apply(lambda r: "Count" if pandas.isnull(r["Embedding size"]) else "Predict", axis=1)

    return regression_df


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
