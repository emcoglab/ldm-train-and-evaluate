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
from os import path

import numpy
from matplotlib import pyplot
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform

from ..core.corpus.indexing import TokenIndexDictionary, FreqDist
from ..core.model.count import PPMIModel, ProbabilityRatioModel, ConditionalProbabilityModel, LogCoOccurrenceCountModel
from ..core.utils.maths import DistanceType
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def main():
    for corpus_metadata in [Preferences.source_corpus_metas[1]]:  # 1 = BBC

        token_index = TokenIndexDictionary.load(corpus_metadata.index_path)
        freq_dist = FreqDist.load(corpus_metadata.freq_dist_path)

        distance_type = DistanceType.cosine

        memory_map = True

        # Not enough room in memory to compute the whole distance matrix, so we'll use the most-frequent words only.
        n_words = 10_000

        for window_radius in Preferences.window_radii:

            logger.info(f"Window radius {window_radius}")

            # COUNT MODELS

            count_models = [
                LogCoOccurrenceCountModel(corpus_metadata, window_radius, token_index),
                ConditionalProbabilityModel(corpus_metadata, window_radius, token_index, freq_dist),
                ProbabilityRatioModel(corpus_metadata, window_radius, token_index, freq_dist),
                PPMIModel(corpus_metadata, window_radius, token_index, freq_dist)
            ]

            for model in count_models:
                model.train(memory_map=memory_map)

                d = pairwise_distances(model.matrix[:n_words, :], metric="cosine", n_jobs=-1)
                numpy.fill_diagonal(d, 0)
                d = squareform(d)

                # Aggregate distances in histogram

                # Define histogram parameters
                data_min = 0  # min possible distance
                data_max = 1  # max possible distance
                n_bins = 250  # number of bins

                bins = numpy.linspace(data_min, data_max, n_bins)

                h, _ = numpy.histogram(d, bins)

                # Save histogram
                bar_width = 1 * (bins[1] - bins[0])
                bar_centres = (bins[:-1] + bins[1:]) / 2
                f, a = pyplot.subplots()
                a.bar(bar_centres, h, align="center", width=bar_width)
                fig_name = f"distance distribution for {model.name} {distance_type.name}.png"
                f.savefig(path.join(Preferences.figures_dir, 'distances', fig_name))

                # release memory
                model.untrain()
            del count_models


if __name__ == '__main__':
    main()
