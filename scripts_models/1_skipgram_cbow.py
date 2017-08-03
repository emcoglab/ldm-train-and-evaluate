import logging
import os
import sys

from core.models.predict import PredictModelSkipGram, PredictModelCBOW
from core.corpus.corpus import CorpusMetadata

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

    weights_dir = "/Users/caiwingfield/vectors/"

    window_radii = [1, 3, 5, 10]
    # These sizes taken from Mandera et al. (2017)
    embedding_sizes = [50, 100, 200, 300, 500]

    for meta in metas:

        for model in ['skip-gram', 'cbow']:

            logger.info(f"Running {model} model")

            save_dir = os.path.join(weights_dir, model)

            for window_radius in window_radii:
                for embedding_size in embedding_sizes:

                    logger.info(f"Working on {meta.name} corpus")

                    weights_path = os.path.join(
                        save_dir,
                        f"{meta.name}_r={window_radius}_s={embedding_size}_{model}.weights")

                    if model == 'skip-gram':
                        predict_model = PredictModelSkipGram(
                            corpus_metadata=meta,
                            weights_path=weights_path,
                            window_radius=window_radius,
                            embedding_size=embedding_size)
                    elif model == 'cbow':
                        predict_model = PredictModelCBOW(
                            corpus_metadata=meta,
                            weights_path=weights_path,
                            window_radius=window_radius,
                            embedding_size=embedding_size)
                    else:
                        raise NotImplementedError()

                    predict_model.build_and_run()

                    logger.info(f"For corpus {meta.name}, "
                                f"model {model}, "
                                f"radius {window_radius}, "
                                f"embedding size {embedding_size}:")
                    logger.info(predict_model.model.most_similar(positive=['woman', 'king'], negative=['man'], topn=4))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
