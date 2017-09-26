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


class SppData(object):
    """
    Semantic Priming Project data.
    """

    def __init__(self):
        # Backing for self.prime_target_data
        self._prime_target_data: pandas.DataFrame = None

    @property
    def prime_target_data(self) -> pandas.DataFrame:
        if self._prime_target_data is None:
            if self._could_load:
                logger.info("Loading previously saved data")
                self._load()
            else:
                logger.info("Loading data from source")
                self._load_from_source()
                # If loading from source, re-save a quick-load copy
                self._save()

        assert self._prime_target_data is not None
        return self._prime_target_data

    def _load(self):
        """
        Load previously saved data.
        """
        with open(Preferences.spp_path, mode="rb") as spp_file:
            self._data = pickle.load(spp_file)

    def _save(self):
        """
        Save data.
        """
        assert self._data is not None
        with open(Preferences.spp_path, mode="wb") as spp_file:
            pickle.dump(self._data, spp_file)

    @property
    def _could_load(self) -> bool:
        """
        Whether data has been previously saved.
        """
        return os.path.isfile(Preferences.spp_path)

    def _load_from_source(self):
        """
        Load data from excel file, dealing with errors in source material.
        """
        xls = pandas.ExcelFile(Preferences.spp_path_xls)
        prime_target_data = xls.parse("Prime-Target Data")

        prime_target_data: pandas.DataFrame = prime_target_data.copy()

        # Convert all to lower case
        prime_target_data["TargetWord"] = prime_target_data["TargetWord"].str.lower()
        prime_target_data["PrimeWord"] = prime_target_data["PrimeWord"].str.lower()

        self._prime_target_data = prime_target_data


class PrimingRegressionTester(object):
    """
    Tests model predictions against a battery of tests, by including model distances as a
    regressor.
    """
    # TODO
    pass
