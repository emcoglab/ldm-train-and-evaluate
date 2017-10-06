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

from ..core.utils.maths import levenshtein_distance
from ..core.evaluation.priming import SppData
from ..core.utils.logging import log_message, date_format
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


# Add prime–target Levenshtein distance
def levenshtein_distance_local(word_pair):
    word_1, word_2 = word_pair
    return levenshtein_distance(word_1, word_2)


def main():
    spp_data = SppData()

    # Add Elexicon predictors

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
        add_elexicon_predictor(spp_data, elexicon_dataframe, predictor_name, prime_or_target="Prime")
        add_elexicon_predictor(spp_data, elexicon_dataframe, predictor_name, prime_or_target="Target")

    # Add prime-target Levenschtein distance

    # Add levenshtein distance column to data frame
    levenshtein_column_name = "PrimeTarget_OrthLD"
    if spp_data.predictor_exists_with_name(levenshtein_column_name):
        logger.info("Levenshtein-distance predictor already added to SPP data.")
    else:
        logger.info("Adding Levenshtein-distance predictor to SPP data.")
        levenshtein_column = spp_data.dataframe[["PrimeWord", "TargetWord"]].copy()
        levenshtein_column[levenshtein_column_name] = levenshtein_column[["PrimeWord", "TargetWord"]].apply(levenshtein_distance_local, axis=1)

        spp_data.add_word_pair_keyed_predictor(levenshtein_column)

    # Save it out for more processing by R or whatever
    spp_data.export_csv()


def add_elexicon_predictor(spp_data: SppData,
                           elexicon_dataframe: pandas.DataFrame,
                           predictor_name: str,
                           prime_or_target: str):

    assert(prime_or_target in ["Prime", "Target"])

    # elex_prime_<predictor_name> or
    # elex_target_<predictor_name>
    new_predictor_name = f"elex_{prime_or_target.lower()}_" + predictor_name

    # PrimeWord or
    # TargetWord
    key_name = f"{prime_or_target}Word"

    # Don't bother training the model until we know we need it
    if spp_data.predictor_exists_with_name(new_predictor_name):
        logger.info(f"Elexicon predictor '{new_predictor_name}' already added to SPP data.")
    else:

        logger.info(f"Adding Elexicon predictor '{new_predictor_name} to SPP data.")

        # Dataframe with two columns: 'Word', [predictor_name]
        predictor = elexicon_dataframe[["Word", predictor_name]]

        # We'll join on PrimeWord first
        predictor = predictor.rename(columns={
            "Word": key_name,
            predictor_name: new_predictor_name
        })

        spp_data.add_word_keyed_predictor(predictor, key_name, new_predictor_name)


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
