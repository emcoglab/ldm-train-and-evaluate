from ..core.corpus.corpus import CorpusMetadata


class Preferences(object):
    """
    Global preferences for models.
    """

    # We will test models with windows of each of these radii
    window_radii = [1, 3, 5, 10]

    # The final locations of the processed corpora
    source_corpus_metas = [
        # CorpusMetadata(
        #     name="toy",
        #     path="/Users/caiwingfield/corpora/toy-corpus/toy.corpus",
        #     index_path="/Users/caiwingfield/vectors/indexes/toy.index"),
        CorpusMetadata(
            name="BBC",
            path="/Users/caiwingfield/corpora/BBC/4 Tokenised/BBC.corpus",
            info_path="/Users/caiwingfield/corpora/BBC/4.1 info",
            index_path="/Users/caiwingfield/vectors/indexes/BBC.index"),
        CorpusMetadata(
            name="BNC",
            path="/Users/caiwingfield/corpora/BNC/2 Tokenised/BNC.corpus",
            info_path="/Users/caiwingfield/corpora/BNC/2.1 info",
            index_path="/Users/caiwingfield/vectors/indexes/BNC.index"),
        CorpusMetadata(
            name="UKWAC",
            path="/Users/caiwingfield/corpora/UKWAC/3 Tokenised/UKWAC.corpus",
            info_path="/Users/caiwingfield/corpora/UKWAC/3.1 info",
            index_path="/Users/caiwingfield/vectors/indexes/UKWAC.index")]
