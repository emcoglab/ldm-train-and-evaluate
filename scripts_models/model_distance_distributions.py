"""
===========================
Produce histograms of model distance distributions.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2018
---------------------------
"""
import logging

import numpy
from matplotlib import pyplot
from os import path

from ..core.corpus.indexing import TokenIndexDictionary, FreqDist
from ..core.model.count import PPMIModel
from ..core.utils.maths import DistanceType
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def main():
    for corpus_metadata in Preferences.source_corpus_metas:

        token_index = TokenIndexDictionary.load(corpus_metadata.index_path)
        freq_dist = FreqDist.load(corpus_metadata.freq_dist_path)

        distance_type = DistanceType.cosine

        for window_radius in Preferences.window_radii:

            logger.info(f"Window radius {window_radius}")

            # COUNT MODELS

            count_models = [
                # LogCoOccurrenceCountModel(corpus_metadata, window_radius, token_index),
                # ConditionalProbabilityModel(corpus_metadata, window_radius, token_index, freq_dist),
                # ProbabilityRatioModel(corpus_metadata, window_radius, token_index, freq_dist),
                PPMIModel(corpus_metadata, window_radius, token_index, freq_dist)
            ]

            for model in count_models:
                logger.info(f"Training model {model.name}")
                model.train(memory_map=True)

                # Aggregate distances in histogram

                # Define histogram parameters
                data_min = 0  # min possible distance
                data_max = 2  # max possible distance
                n_bins = 50  # number of bins

                bins = numpy.linspace(data_min, data_max, n_bins)

                overall_histogram = numpy.zeros(n_bins - 1, dtype='int32')

                n_tokens = len(model.token_indices.tokens)

                # Accumulate histogram
                for i1, word_1 in enumerate(model.token_indices.tokens):
                    logger.info(f"Working on word {i1+1} of {n_tokens}")
                    distances_this_word = [model.distance_between(word_1, word_2, distance_type)
                                           for word_2 in model.token_indices.tokens
                                           if not word_1 == word_2]
                    histogram_this_word, _ = numpy.histogram(distances_this_word, bins)
                    overall_histogram += histogram_this_word

                # Save histogram
                bar_width = 1 * (bins[1] - bins[0])
                bar_centres = (bins[:-1] + bins[1:]) / 2
                f, a = pyplot.subplots()
                a.bar(bar_centres, overall_histogram, align="center", width=bar_width)
                fig_name = f"distance distribution for {model.name} {distance_type.name}.png"
                f.savefig(path.join(Preferences.figures_dir, fig_name))

                # release memory
                model.untrain()
            del count_models


if __name__ == '__main__':
    main()
