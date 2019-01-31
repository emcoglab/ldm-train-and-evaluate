"""
===========================
A script to look at the differences in distance distributions between related and unrelated words in the SPP dataset.
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
import sys
from os import path

from matplotlib import pyplot
from seaborn import distplot

from ..ldm.corpus.indexing import FreqDist
from ..ldm.evaluation.regression import SppData
from ..ldm.model.count import LogCoOccurrenceCountModel, ConditionalProbabilityModel, ProbabilityRatioModel, PPMIModel
from ..ldm.utils.logging import log_message, date_format
from ..ldm.utils.maths import DistanceType
from ..ldm.preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def main():

    spp_data = SppData()

    for corpus in Preferences.source_corpus_metas:
        figures_dir = path.join(Preferences.figures_dir, "priming", "(un)related distances")

        freq_dist: FreqDist = FreqDist.load(corpus.freq_dist_path)

        for radius in Preferences.window_radii:
            for distance_type in DistanceType:
                models = [
                    LogCoOccurrenceCountModel(corpus, radius, freq_dist),
                    ConditionalProbabilityModel(corpus, radius, freq_dist),
                    ProbabilityRatioModel(corpus, radius, freq_dist),
                    PPMIModel(corpus, radius, freq_dist)
                ]
                for model in models:

                    # Load SPP data
                    first_assoc_prime_data = spp_data.dataframe.query('PrimeType == "first_associate"')
                    first_unrel_prime_data = spp_data.dataframe.query('PrimeType == "first_unrelated"')

                    rel_dist = first_assoc_prime_data[spp_data.predictor_name_for_model(model, distance_type, False)].dropna()
                    rel_dist.name = "First related"
                    unrel_dist = first_unrel_prime_data[spp_data.predictor_name_for_model(model, distance_type, False)].dropna()
                    unrel_dist.name = "First unrelated"

                    fig, ax = pyplot.subplots()
                    for dist in [rel_dist, unrel_dist]:
                        distplot(dist, ax=ax,
                                 # hist=False,
                                 label=dist.name,
                                 kde_kws={"label": dist.name})

                    distribution_filename = f"(un)rel distance distributions {model.name} {distance_type.name}.png"

                    logger.info(f"Saving figure to {distribution_filename}...")
                    pyplot.savefig(path.join(figures_dir, distribution_filename))
                    pyplot.close(fig)


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
