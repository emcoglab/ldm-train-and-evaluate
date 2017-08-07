import logging
import os
import sys

from ..core.model.predict import PredictModel
from ..preferences.preferences import Preferences
from ..core.model.base import VectorSpaceModel

logger = logging.getLogger(__name__)


def main():

    weights_dir = "/Users/caiwingfield/vectors/"

    for meta in Preferences.source_corpus_metas:

        for model_type in VectorSpaceModel.Type:

            logger.info(f"Running {model_type} model")

            save_dir = os.path.join(weights_dir, model_type.slug)

            for window_radius in Preferences.window_radii:
                for embedding_size in Preferences.predict_embedding_sizes:

                    weights_path = os.path.join(
                        save_dir,
                        f"{meta.name}_r={window_radius}_s={embedding_size}_{model_type.slug}.weights")

                    # Skip files we've already done
                    if os.path.isfile(weights_path):
                        logger.info(f"Skipping size-{embedding_size} {model_type.name} model from {meta.name} corpus, "
                                    f"window radius {window_radius}")
                        continue

                    logger.info(f"Building size-{embedding_size} {model_type.name} model from {meta.name} corpus, "
                                f"window radius {window_radius}")

                    predict_model = PredictModel(
                        model_type=model_type,
                        corpus=meta,
                        vector_save_path=weights_path,
                        window_radius=window_radius,
                        embedding_size=embedding_size)

                    predict_model.train()
                    predict_model.save()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
