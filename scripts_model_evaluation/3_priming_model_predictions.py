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

from ..core.utils.maths import DistanceType
from ..core.evaluation.priming import SppData
from ..core.model.count import LogNgramModel, ConditionalProbabilityModel, ProbabilityRatioModel, PPMIModel
from ..core.model.predict import SkipGramModel, CbowModel
from ..core.utils.indexing import TokenIndexDictionary
from ..core.corpus.distribution import FreqDist
from ..core.utils.logging import log_message, date_format
from ..preferences.preferences import Preferences


logger = logging.getLogger(__name__)


def main():

    spp_data = SppData()

    for corpus_metadata in Preferences.source_corpus_metas:

        token_index = TokenIndexDictionary.load(corpus_metadata.index_path)
        freq_dist = FreqDist.load(corpus_metadata.freq_dist_path)

        for window_radius in Preferences.window_radii:

            # COUNT MODELS

            count_models = [
                LogNgramModel(corpus_metadata, window_radius, token_index),
                ConditionalProbabilityModel(corpus_metadata, window_radius, token_index, freq_dist),
                ProbabilityRatioModel(corpus_metadata, window_radius, token_index, freq_dist),
                PPMIModel(corpus_metadata, window_radius, token_index, freq_dist)
            ]

            for model in count_models:
                add_predictors_for_model(model, spp_data)

            # PREDICT MODELS

            for embedding_size in Preferences.predict_embedding_sizes:

                predict_models = [
                    SkipGramModel(corpus_metadata, window_radius, embedding_size),
                    CbowModel(corpus_metadata, window_radius, embedding_size)
                ]

                for model in predict_models:
                    add_predictors_for_model(model, spp_data)

    spp_data.export_csv()


def add_predictors_for_model(model, spp_data):

    for distance_type in DistanceType:

        # Don't bother training the model until we know we need it
        if spp_data.predictor_added(model, distance_type):
            logger.info(f"Predictor for '{model.name}' using '{distance_type.name}' already added to SPP data.")
            continue

        logger.info(f"Adding model predictor for '{model.name}' using '{distance_type.name}' to SPP data.")
        model.train()

        spp_data.add_model_predictor(model, distance_type)


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
