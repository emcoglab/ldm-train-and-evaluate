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

import pandas

from ..core.evaluation.priming import SppData
from ..core.utils.logging import log_message, date_format
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def main():
    spp_data = SppData()

    elexicon_dataframe: pandas.DataFrame = pandas.read_csv(Preferences.spp_elexicon_csv, header=0, encoding="utf-8")

    predictors_to_add = {
        # elexicon name : # new name
        "Log_Freq_HAL"  : "elex_Log_Freq_HAL",
        "OLD"           : "elex_OLD20"
    }

    for predictor_key in predictors_to_add.keys():

        predictor_name = predictors_to_add[predictor_key]

        # Don't bother training the model until we know we need it
        if spp_data.predictor_added_with_name(predictor_name):
            logger.info(f"Elexicon predictor '{predictor_name}' already added to SPP data.")
            continue

        predictor = elexicon_dataframe[predictor_key]

        spp_data.add_word_keyed_predictor(predictor, predictor_name)

    spp_data.export_csv()


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
