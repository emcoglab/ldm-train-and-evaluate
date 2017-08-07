import logging
import sys

from ..core.utils.maths import Distance
from ..core.corpus.corpus import CorpusMetadata
from ..core.model.count import CountModel
from ..core.model.base import VectorSpaceModel

logger = logging.getLogger()


def main():
    # TODO: Should work for vectors from all model types
    model_type = VectorSpaceModel.Type.ngram
    matrix_dir = "/Users/caiwingfield/vectors/ngram"

    # TODO: Should work for all Preferences.window_radii
    window_radius = 1

    # TODO: Should work for all corpora
    corpus_metadata = CorpusMetadata(
        name="BNC",
        path="/Users/caiwingfield/corpora/BNC/2 Tokenised/BNC.corpus",
        info_path="/Users/caiwingfield/corpora/BNC/2.1 info",
        index_path="/Users/caiwingfield/vectors/indexes/BNC.index")

    model = CountModel(
        model_type=model_type,
        corpus_meta=corpus_metadata,
        vector_save_path=matrix_dir,
        window_radius=window_radius
    )

    model.load()

    w = "frog"
    n = 10
    d = Distance.Type.Euclidean
    neighbours = model.nearest_neighbours(w, d, n)
    logger.info(f"{n} nearest neighbours to {w}: {neighbours}")


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
