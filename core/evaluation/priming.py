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

import pandas

from ...preferences.preferences import Preferences


logger = logging.getLogger(__name__)


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
            logger.info("Loading data from source")
            self._data = self._load_from_source()
        return self._data

    def _load_from_source(self):
        """
        Load data from excel file, dealing with errors in source material.
        """
        data = pandas.ExcelFile(Preferences.spp_naming_path).parse('Sheet1')

        # Only keep headings which will be useful to us
        data = data[["Subject", "Session", "Trial", "prime", "primecond", "target", "target.RT", "target.ACC"]]

        # Remove empty rows
        data = data[pandas.notnull(data["target"])]
        data = data[pandas.notnull(data["prime"])]

        # Convert all words to lower case
        data["target"] = data["target"].str.lower()
        data["prime"] = data["prime"].str.lower()

        # Ignore rows containing spelling mistakes
        for spelling_error in ["definiton", "lightening", "peonle", "pice"]:
            data = data[~data["target"].str == spelling_error]
            data = data[~data["prime"].str == spelling_error]

        # Remove words containing '
        data = data[~data["target"].str.contains("'")]
        data = data[~data["prime"].str.contains("'")]

        n_targets = len({data["target"].values})
        n_primes = len({data["prime"].values})

        # This is the reported number of targets, so we need to find the discrepancy
        assert (n_targets <= 1661)
        assert (n_primes <= 1661)

        return data


class PrimingRegressionTester(object):
    """
    Tests model predictions against a battery of tests, by including model distances as a
    regressor.
    """
    # TODO
    pass
