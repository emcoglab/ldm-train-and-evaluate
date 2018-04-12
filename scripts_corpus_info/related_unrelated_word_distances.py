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

from matplotlib import pyplot
from seaborn import distplot

from ..preferences.preferences import Preferences
from ..core.corpus.indexing import TokenIndexDictionary, FreqDist
from ..core.utils.maths import DistanceType
from ..core.evaluation.regression import SppData
from ..core.model.count import LogCoOccurrenceCountModel
from ..core.utils.logging import log_message, date_format

logger = logging.getLogger(__name__)


def main():

    # Load model
    corpus = Preferences.source_corpus_metas[1]  # 1: BBC
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    token_index = TokenIndexDictionary.from_freqdist(freq_dist)
    model = LogCoOccurrenceCountModel(corpus, 5, token_index)

    # Load SPP data
    spp_data = SppData()
    first_assoc_prime_data = spp_data.dataframe.query('PrimeType == "first_associate"')
    first_unrel_prime_data = spp_data.dataframe.query('PrimeType == "first_unrelated"')

    rel_dist = first_assoc_prime_data[spp_data.predictor_name_for_model(model, DistanceType.cosine, False)].dropna()
    rel_dist.name = "First related"
    unrel_dist = first_unrel_prime_data[spp_data.predictor_name_for_model(model, DistanceType.cosine, False)].dropna()
    unrel_dist.name = "First unrelated"

    fig, ax = pyplot.subplots()
    for dist in [rel_dist, unrel_dist]:
        distplot(dist, ax=ax,
                 # hist=False,
                 label=dist.name,
                 kde_kws={"label": dist.name})

    pyplot.savefig("/Users/caiwingfield/Desktop/dists.png")

if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
