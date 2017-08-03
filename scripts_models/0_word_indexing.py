import logging
import os
import pickle
import sys

from .preferences import Preferences
from ..core.utils.indexing import TokenIndexDictionary

logger = logging.getLogger(__name__)


def main():
    index_dir = "/Users/caiwingfield/vectors/indexes"

    for meta in Preferences.source_corpus_metas:
        logger.info(f"Producing word index dictionaries for {meta.name} corpus")

        # TODO: this file name should be writte in script_corpus_info.frequency_distribution,
        # TODO: when it's redone as a numbered script
        fdist_path = os.path.join(meta.info_path, f"Frequency distribution {meta.name}.corpus.pickle")

        with open(fdist_path, mode="rb") as freq_dist_file:
            # TODO: Don't use pickle
            freq_dist = pickle.load(freq_dist_file)

        token_index = TokenIndexDictionary.from_freqdist(freq_dist)

        token_index.save(os.path.join(index_dir, f"{meta.name}.index"))


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
