"""
===========================
Evaluate using word similarity judgements.
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

import pandas
from pandas import DataFrame

from .common_output.figures import compare_param_values_bf
from ..core.evaluation.association import SimlexSimilarity, WordsimSimilarity, WordsimRelatedness, MenSimilarity, \
    AssociationResults, ColourEmotionAssociation, ThematicRelatedness
from ..core.utils.logging import log_message, date_format
from ..core.utils.maths import DistanceType
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


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
    # We make an artificial distinction between similarity data
    # and similarity-based association norms,
    # but we treat them both the same
    for artificial_distinction in ["Similarity", "Norms"]:

        figures_base_dir = os.path.join(Preferences.figures_dir, artificial_distinction.lower())

        if artificial_distinction == "Similarity":
            test_names = [SimlexSimilarity().name, WordsimSimilarity().name, WordsimRelatedness().name, MenSimilarity().name]
        elif artificial_distinction == "Norms":
            test_names = [ColourEmotionAssociation().name, ThematicRelatedness().name, ThematicRelatedness(only_use_response=1).name]
        else:
            raise ValueError()

        association_df = AssociationResults().load().data

        association_df["Model category"] = association_df.apply(lambda r: "Count" if pandas.isnull(r["Embedding size"]) else "Predict", axis=1)

        compare_param_values_bf(
            parameter_name="Radius",
            test_results=association_df,
            bf_statistic_name="B10 approx",
            figures_base_dir=figures_base_dir,
            name_prefix=artificial_distinction,
            key_column_name="Test name",
            key_column_values=test_names,
            parameter_values=Preferences.window_radii,
            model_name_func=model_name_without_radius
        )
        compare_param_values_bf(
            parameter_name="Embedding size",
            test_results=association_df,
            bf_statistic_name="B10 approx",
            figures_base_dir=figures_base_dir,
            name_prefix=artificial_distinction,
            key_column_name="Test name",
            key_column_values=test_names,
            parameter_values=Preferences.predict_embedding_sizes,
            model_name_func=model_name_without_embedding_size,
            row_filter=predict_models_only
        )
        compare_param_values_bf(
            parameter_name="Distance type",
            test_results=association_df,
            bf_statistic_name="B10 approx",
            figures_base_dir=figures_base_dir,
            name_prefix=artificial_distinction,
            key_column_name="Test name",
            key_column_values=test_names,
            parameter_values=[d.name for d in DistanceType],
            model_name_func=model_name_without_distance
        )


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
