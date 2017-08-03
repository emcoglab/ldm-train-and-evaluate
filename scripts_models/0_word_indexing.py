import logging
import os
import sys

from .preferences import Preferences
from ..core.corpus.distribution import FreqDist
from ..core.utils.indexing import TokenIndexDictionary

logger = logging.getLogger(__name__)


def main():
    index_dir = "/Users/caiwingfield/vectors/indexes"

    for meta in Preferences.source_corpus_metas:
        logger.info(f"Producing word index dictionaries for {meta.name} corpus")

        # TODO: this file name should be written in script_corpus_info.frequency_distribution,
        # TODO: when it's redone as a numbered script
        freq_dist = FreqDist.load(os.path.join(meta.info_path, f"Frequency distribution {meta.name}.corpus.pickle"))

        token_index = TokenIndexDictionary.from_freqdist(freq_dist)

        token_index.save(os.path.join(index_dir, f"{meta.name}.index"))


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
