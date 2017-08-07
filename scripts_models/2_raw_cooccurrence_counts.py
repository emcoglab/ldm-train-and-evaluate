import logging
import sys

from ..core.model.count import CountModel
from ..core.model.base import VectorSpaceModel
from ..core.utils.indexing import TokenIndexDictionary
from ..preferences.preferences import Preferences

logger = logging.getLogger()


def main():
    for meta in Preferences.source_corpus_metas:

        token_indices = TokenIndexDictionary.load(meta.index_path)

        for radius in range(1, max(Preferences.window_radii) + 1):
            model = CountModel(
                model_type=VectorSpaceModel.Type.ngram_unsummed,
                corpus_metadata=meta,
                vector_save_path="/Users/caiwingfield/vectors/ngram_unsummed",
                window_radius=radius,
                token_indices=token_indices)

            model.train()


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
