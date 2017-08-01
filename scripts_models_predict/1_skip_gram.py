import logging
import sys
import os

from enum import Enum

import gensim

from ..core.corpus import BatchedCorpus, CorpusMetaData


logger = logging.getLogger(__name__)


class PredictModelType(Enum):
    cbow = 0
    skip_gram = 1


def main():

    predict_model_type = PredictModelType.skip_gram

    # Switch on predict model type
    if predict_model_type is PredictModelType.skip_gram:
        model_subdir = 'skip-gram'
        model_name = 'Skip-gram'
        sg = 1
    elif predict_model_type is PredictModelType.cbow:
        model_subdir = 'cbow'
        model_name = 'CBOW'
        sg = 0
    else:
        raise Exception()

    metas = [
        dict(
            corpus=CorpusMetaData(
                name="BBC",
                path="/Users/caiwingfield/corpora/BBC/4 Tokenised/BBC.corpus"),
            weights_save=f"/Users/caiwingfield/vectors/{model_subdir}/BBC.skipgram"),
        dict(
            corpus=CorpusMetaData(
                name="BNC",
                path="/Users/caiwingfield/corpora/BNC/2 Tokenised/BNC.corpus"),
            weights_save=f"/Users/caiwingfield/vectors/{model_subdir}/BNC.skipgram"),
        dict(
            corpus=CorpusMetaData(
                name="UKWAC",
                path="/Users/caiwingfield/corpora/UKWAC/3 Tokenised/UKWAC.corpus"),
            weights_save=f"/Users/caiwingfield/vectors/{model_subdir}/UKWAC.skipgram")
    ]

    for meta in metas:

        logger.info(f"Running on corpus {meta['corpus'].name}")

        if not os.path.isfile(meta['weights_save']):

            logger.info(f"Training {model_name} model")

            # TODO: what size to use?
            embedding_dims      = 100
            # TODO: run at different window radii
            window_radius       = 5
            ignorable_frequency = 1

            # TODO: does using disjoint "sentences" here lead to unpleasant edge effects?
            corpus = BatchedCorpus(filename=meta['corpus'].path, batch_size=10)

            model = gensim.models.Word2Vec(
                sentences=corpus,
                size=embedding_dims,
                window=window_radius,
                min_count=ignorable_frequency,
                sg=sg,
                workers=4)

            model.save(meta['weights_save'])

        else:
            logger.info(f"Loading pre-trained {model_name} model")
            model = gensim.models.Word2Vec.load(meta['weights_save'])

        logger.info(f"For corpus {meta['corpus'].name}:")
        logger.info(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=4))

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
