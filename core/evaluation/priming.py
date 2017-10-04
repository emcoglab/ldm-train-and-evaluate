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
        with open(Preferences.spp_path_pickle, mode="rb") as spp_file:
            return pickle.load(spp_file)

    def _save(self):
        """
        Save and overwrite data.
        """
        assert self._all_data is not None
        with open(Preferences.spp_path_pickle, mode="wb") as spp_file:
            pickle.dump(self._all_data, spp_file)

    def export_csv(self):
        """
        Export the current dataframe as a csv.
        """
        assert self._all_data is not None
        results_csv_path = os.path.join(Preferences.spp_results_dir, "model_predictors.csv")
        with open(results_csv_path, mode="w") as spp_file:
            self.dataframe.to_csv(spp_file)

    @property
    def _could_load(self) -> bool:
        """
        Whether data has been previously saved.
        """
        return os.path.isfile(Preferences.spp_path_pickle)

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

    @property
    def vocabulary(self) -> Set[str]:
        """
        The set of words used in the SPP data.
        """
        vocab: set = set()

        vocab = vocab.union(set(self.dataframe["PrimeWord"]))
        vocab = vocab.union(set(self.dataframe["TargetWord"]))

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
        for word in self.vocabulary:
            if not model.contains_word(word):
                missing_word_list.append(word)

        return sorted([w for w in self.vocabulary if not model.contains_word(w)])

    def predictor_added_with_name(self, predictor_name: str) -> bool:
        """
        Whether the named predictor is already added.
        """
        return self.dataframe.keys().contains(predictor_name)

    @staticmethod
    def predictor_name_for_model(model: VectorSemanticModel, distance_type: DistanceType) -> str:
        return f"{model.name}_{distance_type.name}"

    def add_model_predictor(self, model: VectorSemanticModel, distance_type: DistanceType):
        """
        Adds a data column containing predictors from a semantic model.
        """

        predictor_name = self.predictor_name_for_model(model, distance_type)

        # Skip existing predictors
        if self.predictor_added_with_name(predictor_name):
            logger.info(f"'{predictor_name}' already added")

        else:
            logger.info(f"Adding '{predictor_name}' model")

            # In case we one of the words doesn't exist in the corpus, we just want missing data
            def model_distance_or_none(word_pair):
                word_1, word_2 = word_pair
                try:
                    return model.distance_between(word_1, word_2, distance_type)
                except WordNotFoundError as er:
                    logger.warning(er.args[0])
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

    def add_word_keyed_predictor(self, predictor: pandas.DataFrame, predictor_name: str):
        """
        Adds a pair of predictor word-keyed predictor columns,
        one for the prime word and the other for target word.
        """

        target_predictor_name = 'elex_target_' + predictor_name
        prime_predictor_name = 'elex_prime_' + predictor_name

        # Rename words column in predictor for the target merge
        if not self.predictor_added_with_name(target_predictor_name):
            predictor_renamed = predictor.rename(columns={'Word' : 'TargetWord', predictor_name : target_predictor_name})
            self._all_data = pandas.merge(self.dataframe, predictor_renamed, on="TargetWord", how="left")

        # Rename words column in predictor for the prime merge
        if not self.predictor_added_with_name(prime_predictor_name):
            predictor_renamed = predictor.rename(columns={'Word' : 'PrimeWord', predictor_name : prime_predictor_name})
            self._all_data = pandas.merge(self.dataframe, predictor_renamed, on="PrimeWord", how="left")

        # Add model to list of current models
        self.model_predictor_names.append(target_predictor_name)
        self.model_predictor_names.append(prime_predictor_name)

        # Save in current state
        self._save()

    def add_word_pair_keyed_predictor(self, predictor: pandas.DataFrame):
        """
        Adds a predictor column keyed from a prime-target pair.
        """
        # TODO
        raise NotImplementedError()
