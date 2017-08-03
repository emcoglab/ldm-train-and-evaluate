import logging
import os
import sys

from ..core.models.predict import PredictModelCBOW
from ..core.corpus.corpus import CorpusMetadata

logger = logging.getLogger(__name__)


def main():
    metas = [
        CorpusMetadata(
            name="BBC",
            path="/Users/caiwingfield/corpora/BBC/4 Tokenised/BBC.corpus"),
        CorpusMetadata(
            name="BNC",
            path="/Users/caiwingfield/corpora/BNC/2 Tokenised/BNC.corpus"),
        CorpusMetadata(
            name="UKWAC",
            path="/Users/caiwingfield/corpora/UKWAC/3 Tokenised/UKWAC.corpus")]

    weights_dir = "/Users/caiwingfield/vectors/cbow/"

    for meta in metas:

        # TODO: verify this list
        window_radii = [1, 3, 5, 10]

        for window_radius in window_radii:

            logger.info(f"Working on {meta.name} corpus")

            weights_path = os.path.join(weights_dir, f"{meta.name}_r={window_radius}_cbow.weights")

            predict_model = PredictModelCBOW(
                corpus_metadata=meta,
                weights_path=weights_path,
                window_radius=window_radius)

            predict_model.build_and_run()

            logger.info(f"For corpus {meta.name}:")
            logger.info(predict_model.model.most_similar(positive=['woman', 'king'], negative=['man'], topn=4))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
