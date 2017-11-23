"""
===========================
Creating tables for output.
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

from pandas import DataFrame

from ...core.utils.maths import DistanceType
from ...preferences.preferences import Preferences


def table_top_n_models(results: DataFrame,
                       top_n: int,
                       key_column_name: str,
                       key_column_values: List[str],
                       test_statistic_name: str,
                       name_prefix: str,
                       distance_type: DistanceType = None):

    summary_dir = Preferences.summary_dir

    table = DataFrame()

    for key_column_value in key_column_values:

        filtered_df: DataFrame = results.copy()
        filtered_df = filtered_df[filtered_df[key_column_name] == key_column_value]

        if distance_type is not None:
            filtered_df = filtered_df[filtered_df["Distance type"] == distance_type.name]

        # Assume that "higher is better", so we want to sort values descending
        top_models = filtered_df.sort_values(test_statistic_name, ascending=False).reset_index(drop=True).head(top_n)

        table = table.append(top_models)

    if distance_type is None:
        file_name = f"{name_prefix}_top_{top_n}_models.csv"
    else:
        file_name = f"{name_prefix}_top_{top_n}_models_{distance_type.name}.csv"

    table.to_csv(os.path.join(summary_dir, file_name), index=False)
