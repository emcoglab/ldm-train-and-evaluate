"""
===========================
Plot model vs data for all similarity tasks.
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
from os import path
from typing import List

from pandas import DataFrame

from ..core.utils.logging import log_message, date_format
from ..core.utils.maths import DistanceType
from ..core.corpus.indexing import TokenIndexDictionary, FreqDist
from ..core.evaluation.association import SimlexSimilarity, WordsimSimilarity, WordsimRelatedness, MenSimilarity, \
    ColourEmotionAssociation, ThematicRelatedness, WordAssociationTest
from ..core.model.base import DistributionalSemanticModel
from ..core.model.count import PPMIModel, LogNgramModel, ConditionalProbabilityModel, ProbabilityRatioModel
from ..core.model.predict import SkipGramModel, CbowModel
from .common_output.figures import model_data_scatter_plot
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def main():

    similarity_judgement_tests: List[WordAssociationTest] = [
        SimlexSimilarity(),
        WordsimSimilarity(),
        WordsimRelatedness(),
        MenSimilarity()
    ]
    norm_tests: List[WordAssociationTest] = [
        ColourEmotionAssociation(),
        ThematicRelatedness(),
        ThematicRelatedness(only_use_response=1)
    ]

    # Assemble models
    models: List[DistributionalSemanticModel]
    for corpus_metadata in Preferences.source_corpus_metas:
        token_index = TokenIndexDictionary.load(corpus_metadata.index_path)
        freq_dist = FreqDist.load(corpus_metadata.freq_dist_path)
        for window_radius in Preferences.window_radii:
            # Count models
            models.extend([
                LogNgramModel(corpus_metadata, window_radius, token_index),
                ConditionalProbabilityModel(corpus_metadata, window_radius, token_index, freq_dist),
                ProbabilityRatioModel(corpus_metadata, window_radius, token_index, freq_dist),
                PPMIModel(corpus_metadata, window_radius, token_index, freq_dist)
            ])
            # Predict models
            for embedding_size in Preferences.predict_embedding_sizes:
                models.extend([
                    SkipGramModel(corpus_metadata, window_radius, embedding_size),
                    CbowModel(corpus_metadata, window_radius, embedding_size)
                ])

    for model in models:
        for distance_type in DistanceType:

            for arbitrary_distinction in ["Similarity", "Norms"]:
                figures_dir = path.join(Preferences.figures_dir, arbitrary_distinction.lower())

                test_battery = similarity_judgement_tests if arbitrary_distinction == "Similarity" else norm_tests

                # Load appropriate data
                combined_transcript: DataFrame = DataFrame()
                for test in test_battery:
                    transcript = DataFrame.from_csv(path.join(Preferences.association_results_dir, f"transcript test={test.name} model={model.name} distance={distance_type.name}"))
                    transcript["Test name"] = test.name

                    combined_transcript = combined_transcript.append(transcript)

                # Make the plot
                model_data_scatter_plot(
                    transcript=combined_transcript,
                    name_prefix=arbitrary_distinction,
                    figures_dir=figures_dir
                )

    del models


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
