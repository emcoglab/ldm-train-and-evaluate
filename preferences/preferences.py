"""
===========================
Global preference classes.
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

from ..core.corpus.corpus import CorpusMetadata


class Preferences(object):
    """
    Global preferences for models.
    """

    # We will test models with windows of each of these radii
    window_radii = [1, 3, 5, 10]

    # For the predict models, we will test a number of different embedding sizes
    # These sizes taken from Mandera et al. (2017)
    predict_embedding_sizes = [50, 100, 200, 300, 500]

    # The base directory for the models to be saved
    model_dir = "/Users/caiwingfield/vectors/"

    # The final locations of the processed corpora
    source_corpus_metas = [
        CorpusMetadata(
            name="BBC",
            path="/Users/caiwingfield/corpora/BBC/4 Tokenised/BBC.corpus",
            freq_dist_path="/Users/caiwingfield/corpora/BBC/4.1 info/frequency_distribution_BBC",
            index_path="/Users/caiwingfield/vectors/indexes/BBC.index"),
        CorpusMetadata(
            name="BNC",
            path="/Users/caiwingfield/corpora/BNC/2 Tokenised/BNC.corpus",
            freq_dist_path="/Users/caiwingfield/corpora/BNC/2.1 info/frequency_distribution_BNC",
            index_path="/Users/caiwingfield/vectors/indexes/BNC.index"),
        CorpusMetadata(
            name="UKWAC",
            path="/Users/caiwingfield/corpora/UKWAC/3 Tokenised/UKWAC.corpus",
            freq_dist_path="/Users/caiwingfield/corpora/UKWAC/3.1 info/frequency_distribution_UKWAC",
            index_path="/Users/caiwingfield/vectors/indexes/UKWAC.index")]
