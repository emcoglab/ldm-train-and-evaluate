import logging
import os
import sys

from .preferences import Preferences
from ..core.models.predict import PredictModel, PredictModelType

logger = logging.getLogger(__name__)


def main():

    weights_dir = "/Users/caiwingfield/vectors/"

    # These sizes taken from Mandera et al. (2017)
    embedding_sizes = [50, 100, 200, 300, 500]

    for meta in Preferences.source_corpus_metas:

        for model_type in PredictModelType:

            logger.info(f"Running {model_type} model")

            save_dir = os.path.join(weights_dir, model_type.slug)

            for window_radius in Preferences.window_radii:
                for embedding_size in embedding_sizes:

                    logger.info(f"Building size-{embedding_size} {model_type} model from {meta.name} corpus, "
                                f"window radius {window_radius}")

                    weights_path = os.path.join(
                        save_dir,
                        f"{meta.name}_r={window_radius}_s={embedding_size}_{model_type}.weights")

                    predict_model = PredictModel(
                        model_type=model_type,
                        corpus_metadata=meta,
                        weights_path=weights_path,
                        window_radius=window_radius,
                        embedding_size=embedding_size)

                    predict_model.build_and_run()

                    logger.info(f"For corpus {meta.name}, "
                                f"model {model_type.name}, "
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
