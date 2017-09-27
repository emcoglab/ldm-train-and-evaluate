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

import pandas
import statsmodels.formula.api as sm

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
        # Backing for self.prime_target_data
        self._all_data: pandas.DataFrame = None

    @property
    def dataframe(self) -> pandas.DataFrame:
        if self._all_data is None:
            if self._could_load:
                logger.info("Loading previously saved SPP data")
                self._load()
            else:
                logger.info("Loading SPP data from source xls file")
                self._load_from_source()
                # If loading from source, re-save a quick-load copy
                self._save()

        assert self._all_data is not None
        return self._all_data

    def _load(self):
        """
        Load previously saved data.
        """
        with open(Preferences.spp_path, mode="rb") as spp_file:
            self._all_data = pickle.load(spp_file)

    def _save(self):
        """
        Save data.
        """
        assert self._all_data is not None
        with open(Preferences.spp_path, mode="wb") as spp_file:
            pickle.dump(self._all_data, spp_file)

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

        self._all_data = prime_target_data


class SppRegressionTester(object):
    """
    Tests model predictions against the SPP data, by including model distances as a
    regressor.
    """
    def __init__(self):
        self._spp_data = SppData()

    def administer_test(self,
                        model: VectorSemanticModel):
        """
        Administers the SPP test against a model
        :param model:
        :return:
        """

        assert model.is_trained

        for distance_type in DistanceType:

            data = self._spp_data

            # In case we one of the words doesn't exist in the corpus, we just want missing data
            def model_distance_or_none(word_1, word_2):
                try:
                    return model.distance_between(word_1, word_2, distance_type)
                except WordNotFoundError as er:
                    logger.warning(er)
                    return None

            # Add model distance column to data frame
            data.dataframe["ModelDistance"] = data.dataframe[
                ["PrimeWord", "TargetWord"]
            ].apply(
                model_distance_or_none,
                axis=1)

            # Baseline linear regression
            baseline = sm.ols('NT_200ms_Z ~ '
                              'TargetLogSubFreq + TargetLength + TargetOrthoN + '
                              'PrimeLogSubFreq + PrimeLength + PrimeOrthoN',
                              data=data.dataframe).fit()

            model_fit = sm.ols('NT_200ms_Z ~ '
                               'TargetLogSubFreq + TargetLength + TargetOrthoN + '
                               'PrimeLogSubFreq + PrimeLength + PrimeOrthoN + '
                               'ModelDistance',
                               data=data.dataframe).fit()

            logger.info(f"Model: {model.name}, \t"
                        f"distance: {distance_type.name}, \t"
                        f"baseline: {baseline.rsquared}, \t"
                        f"model fit: {model_fit.rsquared}")

            # TODO: bayes factor of one over the other
