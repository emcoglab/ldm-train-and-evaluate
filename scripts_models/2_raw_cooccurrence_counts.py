import logging
import sys

from ..core.utils.constants import Chirality
from ..core.model.count import UnsummedNgramCountModel
from ..core.utils.indexing import TokenIndexDictionary
from ..preferences.preferences import Preferences

logger = logging.getLogger()


def main():
    for meta in Preferences.source_corpus_metas:

        token_indices = TokenIndexDictionary.load(meta.index_path)

        for radius in range(1, max(Preferences.window_radii) + 1):
            for chirality in Chirality:
                model = UnsummedNgramCountModel(
                    corpus=meta,
                    save_dir="/Users/caiwingfield/vectors/",
                    window_radius=radius,
                    token_indices=token_indices,
                    chirality=chirality)
                model.train()
                model.save()


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
