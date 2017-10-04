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

    # Make sure the words are lowercase, as we'll use them as merging keys
    elexicon_dataframe['Word'] = elexicon_dataframe['Word'].str.lower()

    predictors_to_add = [
        "LgSUBTLWF",
        "OLD",
        "PLD",
        "NSyll"
    ]

    for predictor_name in predictors_to_add:

        # Don't bother training the model until we know we need it
        if spp_data.predictor_added_with_name(predictor_name):
            logger.info(f"Elexicon predictor '{predictor_name}' already added to SPP data.")
            continue

        # Dataframe with two columns: 'Word', [predictor_name]
        predictor_column = elexicon_dataframe[["Word", predictor_name]]

        spp_data.add_word_keyed_predictor(predictor_column, predictor_name)


    spp_data.export_csv()


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
