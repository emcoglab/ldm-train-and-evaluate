import logging
import sys
import os

import gensim

from ..core.corpus import BatchedCorpus, CorpusMetaData

logger = logging.getLogger(__name__)


def main():

    metas = [
        dict(
            corpus=CorpusMetaData(
                name="BBC",
                path="/Users/caiwingfield/corpora/BBC/4 Tokenised/BBC.corpus"),
            weights_save="/Users/caiwingfield/vectors/skip-gram/BBC.skipgram"),
        dict(
            corpus=CorpusMetaData(
                name="BNC",
                path="/Users/caiwingfield/corpora/BNC/2 Tokenised/BNC.corpus"),
            weights_save="/Users/caiwingfield/vectors/skip-gram/BNC.skipgram"),
        dict(
            corpus=CorpusMetaData(
                name="UKWAC",
                path="/Users/caiwingfield/corpora/UKWAC/3 Tokenised/UKWAC.corpus"),
            weights_save="/Users/caiwingfield/vectors/skip-gram/UKWAC.skipgram")
    ]

    for meta in metas:

        logger.info(f"Running on corpus {meta['corpus'].name}")

        if not os.path.isfile(meta['weights_save']):

            logger.info("Training Skip-gram")

            embedding_dims      = 100
            window_radius       = 5
            ignorable_frequency = 1

            corpus = BatchedCorpus(filename=meta['corpus'].path, batch_size=10)

            # TODO: is this actually skip-gram or cbow?
            model = gensim.models.Word2Vec(
                # TODO: does using disjoing "sentences" here lead to unpleasant edge effects?
                sentences=corpus,
                size=embedding_dims,
                window=window_radius,
                min_count=ignorable_frequency,
                workers=4)

            model.save(meta['weights_save'])

        else:
            logger.info("Loading pre-trained model")
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
