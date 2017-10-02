"""
===========================
Testing against human priming data.
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
import pickle

from typing import List, Set

import pandas

from ..model.base import VectorSemanticModel
from ..utils.maths import DistanceType
from ..utils.exceptions import WordNotFoundError
from ...preferences.preferences import Preferences

logger = logging.getLogger(__name__)


class SppData(object):
    """
    Semantic Priming Project data.
    """

    def __init__(self):

        # self._all_data backs self.dataframe
        # Load data if possible
        if self._could_load:
            logger.info("Loading previously saved SPP data")
            self._all_data = self._load()
        else:
            logger.info("Loading SPP data from source xls file")
            self._all_data = self._load_from_source_xls()
            # If loading from source, re-save a quick-load copy
            self._save()
        assert self._all_data is not None

        # Names of currently added predictor models
        self.model_predictor_names: List[str] = []

    @property
    def dataframe(self) -> pandas.DataFrame:
        return self._all_data

    @classmethod
    def _load(cls) -> pandas.DataFrame:
        """
        Load previously saved data.
        """
        with open(Preferences.spp_path, mode="rb") as spp_file:
            return pickle.load(spp_file)

    def _save(self):
        """
        Save and overwrite data.
        """
        assert self._all_data is not None
        with open(Preferences.spp_path, mode="wb") as spp_file:
            pickle.dump(self._all_data, spp_file)

    def export_csv(self):
        """
        Export the current dataframe as a csv.
        """
        # TODO
        raise NotImplementedError()

    @property
    def _could_load(self) -> bool:
        """
        Whether data has been previously saved.
        """
        return os.path.isfile(Preferences.spp_path)

    @classmethod
    def _load_from_source_xls(cls) -> pandas.DataFrame:
        """
        Load data from excel file, dealing with errors in source material.
        """
        xls = pandas.ExcelFile(Preferences.spp_path_xls)
        prime_target_data = xls.parse("Prime-Target Data")

        prime_target_data: pandas.DataFrame = prime_target_data.copy()

        # Convert all to lower case
        prime_target_data["TargetWord"] = prime_target_data["TargetWord"].str.lower()
        prime_target_data["PrimeWord"] = prime_target_data["PrimeWord"].str.lower()

        return prime_target_data

    def vocabulary(self) -> Set[str]:
        """
        The set of words used in the SPP data.
        """
        vocab: set = set()

        vocab += set(self.dataframe["PrimeWord"])
        vocab += set(self.dataframe["TargetWord"])

        return vocab

    def missing_words(self, model: VectorSemanticModel) -> List[str]:
        """
        The list of SPP words which aren't present in a model.
        :type model: VectorSemanticModel
        :param model: Must be trained.
        :return: List of missing words.
        """
        assert model.is_trained

        missing_word_list = []
        for word in self.vocabulary():
            if not model.contains_word(word):
                missing_word_list.append(word)

        return [w for w in self.vocabulary() if not model.contains_word(w)]

    def add_model_predictor(self, model: VectorSemanticModel, distance_type: DistanceType):
        """
        Adds a data column containing predictors from a semantic model.
        :param distance_type:
        :param model:
        """

        predictor_name = f"{model.name}_{distance_type.name}"

        # Skip existing predictors
        if self.dataframe.names.contains(predictor_name):
            logger.info(f"{predictor_name} already added")

        else:
            logger.info(f"Adding {predictor_name} model")

            # In case we one of the words doesn't exist in the corpus, we just want missing data
            def model_distance_or_none(word_1, word_2):
                try:
                    return model.distance_between(word_1, word_2, distance_type)
                except WordNotFoundError as er:
                    logger.warning(er)
                    return None

            # Add model distance column to data frame
            self.dataframe[predictor_name] = self.dataframe[
                ["PrimeWord", "TargetWord"]
            ].apply(
                model_distance_or_none,
                axis=1)

            # Add model to list of current models
            self.model_predictor_names.append(predictor_name)

            # Save in current state
            self._save()
