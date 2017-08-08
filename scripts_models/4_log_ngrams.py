import logging
import sys

from ..core.model.count import LogNgramModel
from ..core.utils.indexing import TokenIndexDictionary
from ..preferences.preferences import Preferences

logger = logging.getLogger()


def main():

    for meta in Preferences.source_corpus_metas:

        token_indices = TokenIndexDictionary.load(meta.index_path)

        for radius in Preferences.window_radii:

            model = LogNgramModel(meta, "/Users/caiwingfield/vectors/", radius, token_indices)
            model.train()
            model.save()


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
