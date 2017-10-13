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

from abc import ABCMeta, abstractmethod
from typing import List

from ..model.base import VectorSemanticModel
from ..model.predict import PredictVectorModel
from ..utils.maths import DistanceType


class ReportCard(object, metaclass=ABCMeta):
    """
    Description of the results of a battery of tests.
    """

    @classmethod
    @abstractmethod
    def results_dir(cls) -> str:
        """
        Where the results should be saved.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def headings(cls) -> List[str]:
        """
        An ordered list of the headings for each piece of data on the report card.
        """
        raise NotImplementedError()

    def __init__(self):
        self.entries: List[ReportCard.Entry] = []

    def __iadd__(self, other: 'ReportCard'):
        for entry in other.entries:
            self.add_entry(entry)

    def __add__(self, other: 'ReportCard'):
        new = ReportCard()
        for entry in self.entries:
            new.add_entry(entry)
        for entry in other.entries:
            new.add_entry(entry)
        return new

    def add_entry(self, entry: 'ReportCard.Entry'):
        """
        Adds an entry to the report card.
        """
        self.entries.append(entry)

    def save_headers(self, separator: str = ","):
        """
        Saves a CSV file containing headers, if it doesn't already exist.
        """

        csv_filename = " header.csv"

        csv_path = os.path.join(self.results_dir(), csv_filename)

        # Skip it if it exists
        if os.path.isfile(csv_path):
            return

        with open(csv_path, mode="w", encoding="utf-8") as csv_file:

            # Write headings
            csv_file.write(separator.join(self.headings()) + "\n")

    @classmethod
    def saved_with_name(cls, csv_filename: str) -> bool:
        """
        Has a report card been saved with this name?
        """
        return os.path.isfile(
            os.path.join(cls.results_dir(), csv_filename))

    def save_csv(self, csv_filename: str, separator: str = ","):
        """
        Writes records to a CSV, overwriting if it exists.
        :param csv_filename:
        :param separator:
        """
        # Validate filename
        if not csv_filename.endswith(".csv"):
            csv_filename += ".csv"

        csv_path = os.path.join(self.results_dir(), csv_filename)

        with open(csv_path, mode="w", encoding="utf-8") as csv_file:
            for entry in self.entries:
                csv_file.write(separator.join(entry.fields) + "\n")

        # Make sure the headers are saved too
        self.save_headers()

    class Entry(object, metaclass=ABCMeta):
        """
        Description of the results of a single test.
        """

        def __init__(self,
                     test_name: str,
                     model_type_name: str,
                     model: VectorSemanticModel,
                     distance_type: DistanceType):
            self._distance_type = distance_type
            self._model_type_name = model_type_name
            self._window_radius = model.window_radius
            self._corpus_name = model.corpus_meta.name
            self._embedding_size = model.embedding_size if isinstance(model, PredictVectorModel) else None
            self._test_name = test_name

        @property
        @abstractmethod
        def fields(self) -> List[str]:
            raise NotImplementedError()
