"""
===========================
Base classes for results of tests.
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
import pickle

from abc import ABCMeta
from typing import List

import pandas

from ..model.base import VectorSemanticModel
from ..model.predict import PredictVectorModel
from ..utils.maths import DistanceType


class EvaluationResults(metaclass=ABCMeta):
    """
    The results of a model evaluation.
    """

    model_index_column_names = [
       "Test name",
       "Model type",
       "Embedding size",
       "Radius",
       "Distance type",
       "Corpus"
    ]

    def __init__(self,
                 results_column_names: List[str],
                 save_dir: str):

        self._save_path = os.path.join(save_dir, "evaluation_results")
        self._pickle_path = self._save_path + ".pickle"
        self._csv_path = self._save_path + ".csv"

        column_names = self.model_index_column_names + results_column_names

        self.data: pandas.DataFrame = None
        if self._previously_saved():
            self.data = self._load()
            assert set(self.column_names) == set(column_names)
        else:
            self.data = pandas.DataFrame(columns=column_names)
        assert self.data is not None

    @property
    def column_names(self):
        """
        The column names in the results table.
        """
        return self.data.columns.values

    def add_result(self,
                   test_name: str,
                   model: VectorSemanticModel,
                   distance_type: DistanceType,
                   # a dictionary whose keys are the same as the results_column_names
                   result: dict,
                   append_to_model_name: str = None):
        """
        Add a single result.
        """
        # Add model keys to result row
        result["Test name"] = test_name
        # TODO: this is also gross
        result["Model type"] = model.model_type.name + (append_to_model_name if append_to_model_name is not None else "")
        result["Embedding size"] = model.embedding_size if isinstance(model, PredictVectorModel) else None
        result["Radius"] = model.window_radius
        result["Distance type"] = distance_type.name
        result["Corpus"] = model.corpus_meta.name

        assert set(result.keys()) == set(self.column_names)

        # TODO: This is possibly inefficient
        self.data = self.data.append(result, ignore_index=True)

    def extend_with_results(self, other: 'EvaluationResults'):
        """
        Adds a batch of results.
        """
        assert set(self.column_names) == set(other.column_names)
        self.data = self.data.append(other.data, ignore_index=True)

    @classmethod
    def results_exist_for(cls,
                          test_name: str,
                          model: VectorSemanticModel,
                          distance_type: DistanceType,
                          truncate_vectors_at_length: int = None) -> bool:
        """
        Do results exist for this model?
        """
        instance = cls()
        return instance.data[
                   (instance.data["Test name"] == test_name) &
                   # TODO: this is gross
                   (instance.data["Model type"] == (model.model_type.name + (f" ({truncate_vectors_at_length})" if truncate_vectors_at_length is not None else ""))) &
                   ((instance.data["Embedding size"] == model.embedding_size) if isinstance(model, PredictVectorModel) else True) &
                   (instance.data["Radius"] == model.window_radius) &
                   (instance.data["Distance type"] == distance_type.name) &
                   (instance.data["Corpus"] == model.corpus_meta.name)
               ].count() > 0

    def save(self):
        """
        Save (and overwrite) data.
        """
        assert self.data is not None
        with open(self._pickle_path, mode="wb") as data_file:
            pickle.dump(self.data, data_file)
        self._export_csv()

    def _previously_saved(self) -> bool:
        return os.path.isfile(self._pickle_path)

    def _load(self) -> pandas.DataFrame:
        """
        Load previously saved data.
        """
        with open(self._pickle_path, mode="rb") as data_file:
            return pickle.load(data_file)

    def _export_csv(self):
        """
        Export results as a csv.
        """
        with open(self._csv_path, mode="w", encoding="utf-8") as spp_file:
            self.data.to_csv(spp_file)

    def import_csv(self):
        """
        Load from a CSV (only use if the pickle gets lost or corrupted somehow.
        """
        self.data = pandas.read_csv(self._csv_path)
