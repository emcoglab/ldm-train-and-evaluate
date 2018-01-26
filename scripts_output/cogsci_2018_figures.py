"""
===========================
Figures for CogSci 2018 paper.
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

import sys
import logging
from os import path

import seaborn
import numpy
from pandas import DataFrame
from matplotlib import pyplot

from .common_output.figures import yticks_as_percentages
from .common_output.dataframe import add_model_category_column
from ..core.utils.logging import log_message, date_format
from ..core.model.base import DistributionalSemanticModel
from ..core.evaluation.synonym import ToeflTest, EslTest, LbmMcqTest, SynonymResults
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)

FIGURES_BASE_DIR = path.join(Preferences.evaluation_dir, "for publication", "cogsci2018")


def main():

    # Synonym tests

    synonym_results: DataFrame = SynonymResults().load().data
    add_model_category_column(synonym_results)

    for test_name in [ToeflTest().name, EslTest().name, LbmMcqTest().name]:
        test_results: DataFrame = synonym_results[synonym_results["Test name"] == test_name]
        single_violin_plot(results=test_results,
                           key_column_name="Test name",
                           test_statistic_name="Score",
                           name_prefix=test_name,
                           extra_h_line_at=0.25,
                           ticks_as_percentages=True,
                           ylim=(0, 1))


def single_violin_plot(results: DataFrame,
                       key_column_name: str,
                       test_statistic_name: str,
                       name_prefix: str,
                       extra_h_line_at: float=None,
                       ticks_as_percentages: bool=False,
                       ylim=None):

    seaborn.set_style("ticks")

    local_results: DataFrame = results.copy()

    # Don't want to show PPMI (10000)
    # This only applies for synonym tests, but it doesn't cause a problem if it's not present
    local_results = local_results[local_results["Model type"] != "PPMI (10000)"]

    # !!!!
    # NOTE: For the purposes of display, we make all values positive! This should be acknowledged in any legend!
    # !!!!
    local_results[test_statistic_name] = local_results[test_statistic_name].apply(numpy.abs)

    seaborn.set_context(context="paper", font_scale=1)

    grid = seaborn.FacetGrid(
        local_results,
        row=key_column_name,
        hue="Model category",
        hue_order=["N-gram", "Count", "Predict"],
        margin_titles=True,
        size=4
    )

    if extra_h_line_at is not None:
        # Plot the chance line
        grid.map(pyplot.axhline, y=extra_h_line_at, linestyle="solid", color="xkcd:bright red")

    grid.set_xticklabels(rotation=-90)

    if ticks_as_percentages:
        yticks_as_percentages(grid)

    # Plot the bars
    grid.map(
        seaborn.violinplot, "Model type", test_statistic_name,
        # width=10,
        cut=0, inner=None, linewidth=0,
        order=[
            # ngram
            DistributionalSemanticModel.ModelType.log_ngram.name,
            DistributionalSemanticModel.ModelType.probability_ratio_ngram.name,
            DistributionalSemanticModel.ModelType.ppmi_ngram.name,
            # count
            DistributionalSemanticModel.ModelType.log_cooccurrence.name,
            DistributionalSemanticModel.ModelType.conditional_probability.name,
            DistributionalSemanticModel.ModelType.probability_ratio.name,
            DistributionalSemanticModel.ModelType.ppmi.name,
            # predict
            DistributionalSemanticModel.ModelType.skip_gram.name,
            DistributionalSemanticModel.ModelType.cbow.name,
        ]
    )

    grid.map(
        seaborn.swarmplot, "Model type", test_statistic_name,
        marker="o", color="0", size=2,
        order=[
            # ngram
            DistributionalSemanticModel.ModelType.log_ngram.name,
            DistributionalSemanticModel.ModelType.probability_ratio_ngram.name,
            DistributionalSemanticModel.ModelType.ppmi_ngram.name,
            # count
            DistributionalSemanticModel.ModelType.log_cooccurrence.name,
            DistributionalSemanticModel.ModelType.conditional_probability.name,
            DistributionalSemanticModel.ModelType.probability_ratio.name,
            DistributionalSemanticModel.ModelType.ppmi.name,
            # predict
            DistributionalSemanticModel.ModelType.skip_gram.name,
            DistributionalSemanticModel.ModelType.cbow.name,
        ]
    )

    grid.set_ylabels(test_statistic_name)

    if ylim is not None:
        grid.set(ylim=ylim)

    grid.fig.tight_layout()

    pyplot.subplots_adjust(top=0.92)
    grid.fig.suptitle(f"{name_prefix}: Overall model performance")

    figure_name = f"{name_prefix} ({test_statistic_name}).png"

    grid.savefig(path.join(FIGURES_BASE_DIR, figure_name), dpi=300)

    pyplot.close(grid.fig)


if __name__ == '__main__':
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")