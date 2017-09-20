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
import pickle
import os

import pandas

from ...preferences.preferences import Preferences

logger = logging.getLogger(__name__)


class SppItems(object):
    """
    Semantic Priming Project items.
    """

    def __init__(self):
        # Backing for self.items
        self._items: pandas.DataFrame = None

    # @property
    # def items(self) -> pandas.DataFrame:
    #     if self._items is None:
    #         if self._could_load:
    #             logger.info("Loading previously saved data")
    #             self._load()
    #         else:
    #             logger.info("Loading data from source")
    #             self._load_from_source()
    #             # Rename the columns to something intelligible
    #             self._rename_columns()
    #             # If loading from source, re-save a quick-load copy
    #             self._save()
    #     assert self._items is not None
    #
    #     # Only expose useful headings
    #     return self._items
    #
    # def _load(self):
    #     """
    #     Load previously saved data.
    #     """
    #     with open(Preferences.spp_items_path, mode="rb") as spp_naming_file:
    #         self._data = pickle.load(spp_naming_file)
    #
    # def _save(self):
    #     """
    #     Save data.
    #     """
    #     assert self._data is not None
    #     with open(Preferences.spp_items_path, mode="wb") as spp_naming_file:
    #         pickle.dump(self._data, spp_naming_file)
    #
    # @property
    # def _could_load(self) -> bool:
    #     """
    #     Whether data has been previously saved.
    #     """
    #     return os.path.isfile(Preferences.spp_items_path)

    def _load_from_source(self):
        """
        Load data from excel file, dealing with errors in source material.
        """
        xls = pandas.ExcelFile(Preferences.spp_items_path_xls)
        first_associate = xls.parse("first associate")
        other_associate = xls.parse("other associate")
        unrelated_pairs = xls.parse("unrelated pairs")

        first_associate_prime: pandas.DataFrame = first_associate.copy()[[
            "prime_first associate", "f_length", "f_log subtl freq", "f_orthoN"
        ]]
        other_associate_prime: pandas.DataFrame = other_associate.copy()[[
            "other Assoc", "o_length", "o_log subtl freq", "o_orthoN"
        ]]
        unrelated_first_associate: pandas.DataFrame = unrelated_pairs.copy()[[
            "f_unlreated"  # sic
        ]]
        unrelated_other_associate: pandas.DataFrame = unrelated_pairs.copy()[[
            "o_unrelated"
        ]]
        targets: pandas.DataFrame = first_associate.copy()[[
            "TARGET", "t_length", "t_log subtl freq", "t_orthoN"
        ]]

        # Convert all to lower case
        first_associate_prime["prime_first associate"] = first_associate_prime["prime_first associate"].str.lower()
        targets["TARGET"] = targets["TARGET"].str.lower()
        other_associate_prime["other Assoc"] = other_associate_prime["other Assoc"].str.lower()
        unrelated_first_associate["f_unlreated"] = unrelated_first_associate["f_unlreated"].str.lower()
        unrelated_other_associate["o_unrelated"] = unrelated_other_associate["o_unrelated"].str.lower()

        # Rename columns
        targets.rename(columns={
            "TARGET": "word",
            "t_length": "length",
            "t_log subtl freq": "log frequency (SUBLEX-US)",
            "t_orthoN": "orthographic neighbourhood density"
        }, inplace=True)
        first_associate_prime.rename(columns={
            "prime_first associate": "word",
            "f_length": "length",
            "f_log subtl freq": "log frequency (SUBLEX-US)",
            "f_orthoN": "orthographic neighbourhood density"
        }, inplace=True)
        other_associate_prime.rename(columns={
            "other Assoc": "word",
            "o_length": "length",
            "o_log subtl freq": "log frequency (SUBLEX-US)",
            "o_orthoN": "orthographic neighbourhood density"
        }, inplace=True)
        unrelated_first_associate.rename(columns={
            "f_unlreated": "word"
        }, inplace=True)
        unrelated_other_associate.rename(columns={
            "o_unrelated": "word"
        }, inplace=True)

        # TODO: Deal with typos
        # >>> set(first_associate_prime["word"].values) - set(unrelated_first_associate["f_unlreated"].values)
        # {'bead', 'bleach', 'frequency', 'skillet', 'confidence', 'celsius', 'stretch'}
        # >>> set(unrelated_first_associate["f_unlreated"].values) - set(first_associate_prime["word"].values)
        # {'celcius', 'grade', 'skiller', 'condfidence'}

        # TODO: Combine primes

    def _rename_columns(self):
        """
        Rename columns to something intelligible.
        """
        # TODO
        self._items.rename({
            "old column name 1": "new column name 1",
            "old column name 2": "new column name 2"
        }, inplace=True)


class SppNaming(object):
    """
    Naming data from Semantic Priming Project.
    """

    def __init__(self):
        # Backing for self.data
        self._data: pandas.DataFrame = None

    @property
    def data(self) -> pandas.DataFrame:
        if self._data is None:
            if self._could_load:
                logger.info("Loading previously saved data")
                self._load()
            else:
                logger.info("Loading data from source")
                self._load_from_source()
                # If loading from source, re-save a quick-load copy
                self._save()
        assert self._data is not None

        # Only expose useful headings
        return self._data[[
            "Subject", "Session", "Trial", "prime", "target", "target.RT", "target.ACC"
        ]]

    @property
    def correct_prime_target_pairs(self) -> pandas.DataFrame:
        """
        Prime-Target pairs.
        """
        return self.data.where(self.data["target.ACC"] == 1).groupby(["prime", "target"])

    @property
    def n_targets(self) -> int:
        """
        The number of targets in the dataset.
        """
        return len(set(self.data["target"].values))

    @property
    def n_primes(self) -> int:
        """
        The number of primes in the dataset.
        """
        return len(set(self.data["prime"].values))

    def _load(self):
        """
        Load previously saved data.
        """
        with open(Preferences.spp_naming_path, mode="rb") as spp_naming_file:
            self._data = pickle.load(spp_naming_file)

    def _save(self):
        """
        Save data.
        """
        assert self._data is not None
        with open(Preferences.spp_naming_path, mode="wb") as spp_naming_file:
            pickle.dump(self._data, spp_naming_file)

    @property
    def _could_load(self) -> bool:
        """
        Whether data has been previously saved.
        """
        return os.path.isfile(Preferences.spp_naming_path)

    def _load_from_source(self):
        """
        Load data from excel file, dealing with errors in source material.
        """
        data = pandas.ExcelFile(Preferences.spp_naming_path_xls).parse('Sheet1')

        # Remove empty rows
        data = data[pandas.notnull(data["target"])]
        data = data[pandas.notnull(data["prime"])]

        # Convert all words to lower case
        data["target"] = data["target"].str.lower()
        data["prime"] = data["prime"].str.lower()

        # Ignore rows containing spelling mistakes
        # TODO: "lightening" is a word, but is it the intended one?
        for spelling_error in ["definiton", "lightening", "peonle", "pice"]:
            data = data[data["target"].str.lower() != spelling_error]
            data = data[data["prime"].str.lower() != spelling_error]

        # Remove words containing ', which would be split into two words in our corpus and hence not found
        data = data[~data["target"].str.contains("'")]
        data = data[~data["prime"].str.contains("'")]

        self._data = data


class PrimingRegressionTester(object):
    """
    Tests model predictions against a battery of tests, by including model distances as a
    regressor.
    """
    # TODO
    pass
