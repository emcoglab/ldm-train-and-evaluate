import logging
import os
import pickle
import sys

from ..core.classes import CorpusMetaData
from ..core.indexing import TokenIndexDictionary

logger = logging.getLogger(__name__)


def main():
    fdist_metas = [
        CorpusMetaData(
            name="toy",
            path="/Users/caiwingfield/corpora/toy-corpus/info/Frequency distribution toy.corpus.pickle"
        ),
        CorpusMetaData(
            name="BBC",
            path="/Users/caiwingfield/corpora/BBC/4.1 info/Frequency distribution BBC.corpus.pickle"
        ),
        CorpusMetaData(
            name="BNC",
            path="/Users/caiwingfield/corpora/BNC/2.1 info/Frequency distribution BNC.corpus.pickle"
        ),
        CorpusMetaData(
            name="UKWAC",
            path="/Users/caiwingfield/corpora/UKWAC/3.1 info/Frequency distribution UKWAC.corpus.pickle"
        )]

    index_dir = "/Users/caiwingfield/vectors/indexes"

    for fdist_meta in fdist_metas:
        logger.info(f"Producing word index dictionaries for {fdist_meta.name} corpus")

        with open(fdist_meta.path, mode="rb") as freq_dist_file:
            freq_dist = pickle.load(freq_dist_file)

        token_index = TokenIndexDictionary.from_freqdist(freq_dist)

        token_index.save(os.path.join(index_dir, f"{fdist_meta.name}.index"))


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
