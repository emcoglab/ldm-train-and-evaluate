import logging
import sys

from ..core.models.predict import PredictModelSkipGram
from ..core.corpus.corpus import CorpusMetadata

logger = logging.getLogger(__name__)


def main():
    metas = [
        dict(
            corpus=CorpusMetadata(
                name="BBC",
                path="/Users/caiwingfield/corpora/BBC/4 Tokenised/BBC.corpus"),
            weights_save=f"/Users/caiwingfield/vectors/skip-gram/BBC_skipgram.weights"),
        dict(
            corpus=CorpusMetadata(
                name="BNC",
                path="/Users/caiwingfield/corpora/BNC/2 Tokenised/BNC.corpus"),
            weights_save=f"/Users/caiwingfield/vectors/skip-gram/BNC_skipgram.weights"),
        dict(
            corpus=CorpusMetadata(
                name="UKWAC",
                path="/Users/caiwingfield/corpora/UKWAC/3 Tokenised/UKWAC.corpus"),
            weights_save=f"/Users/caiwingfield/vectors/skip-gram/UKWAC_skipgram.weights")
    ]

    for meta in metas:

        logger.info(f"Working on {meta['corpus'].name} corpus")

        predict_model = PredictModelSkipGram(
            corpus_metadata=meta['corpus'],
            weights_path=meta['weights_save'])

        predict_model.build_and_run()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
