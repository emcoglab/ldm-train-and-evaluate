import logging
import sys

from ..core.models.predict import PredictModelCBOW
from ..core.corpus.corpus import CorpusMetadata

logger = logging.getLogger(__name__)


def main():
    metas = [
        dict(
            corpus=CorpusMetadata(
                name="BBC",
                path="/Users/caiwingfield/corpora/BBC/4 Tokenised/BBC.corpus"),
            weights_save=f"/Users/caiwingfield/vectors/cbow/BBC_cbow.weights"),
        dict(
            corpus=CorpusMetadata(
                name="BNC",
                path="/Users/caiwingfield/corpora/BNC/2 Tokenised/BNC.corpus"),
            weights_save=f"/Users/caiwingfield/vectors/cbow/BNC_cbow.weights"),
        dict(
            corpus=CorpusMetadata(
                name="UKWAC",
                path="/Users/caiwingfield/corpora/UKWAC/3 Tokenised/UKWAC.corpus"),
            weights_save=f"/Users/caiwingfield/vectors/cbow/UKWAC_cbow.weights")
    ]

    for meta in metas:

        logger.info(f"Working on {meta['corpus'].name} corpus")

        predict_model = PredictModelCBOW(
            corpus_metadata=meta['corpus'],
            weights_path=meta['weights_save'])

        predict_model.build_and_run()

        logger.info(f"For corpus {meta['corpus'].name}:")
        logger.info(predict_model.model.most_similar(positive=['woman', 'king'], negative=['man'], topn=4))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
