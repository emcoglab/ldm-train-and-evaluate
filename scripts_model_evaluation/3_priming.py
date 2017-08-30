"""
===========================
Evaluate using priming data.
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
import sys

import pandas as pd
import numpy as np
import xlrd

from ..core.utils.logging import log_message, date_format


logger = logging.getLogger(__name__)


def main():
    xls = pd.ExcelFile("/Users/caiwingfield/evaluation/tests/Semantic priming project/all naming subjects.xlsx")
    df = xls.parse('Sheet1')
    df.where()

    return


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
