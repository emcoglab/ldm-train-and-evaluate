import logging
import sys
import os

import gensim

from ..core.corpus import BatchedCorpus

logger = logging.getLogger(__name__)


def main():

    model_save_filename = "/Users/caiwingfield/vectors/skip-gram/BNC.skipgram"

    if not os.path.isfile(model_save_filename):

        embedding_dims      = 100
        window_radius       = 5
        ignorable_frequency = 1

        corpus = BatchedCorpus(filename="/Users/caiwingfield/corpora/BNC/2 Tokenised/BNC.corpus", batch_size=10)

        model = gensim.models.Word2Vec(
            # TODO: does using disjoing "sentences" here lead to unpleasant edge effects?
            sentences=corpus,
            size=embedding_dims,
            window=window_radius,
            min_count=ignorable_frequency,
            workers=4)

        model.save(model_save_filename)

    else:

        model = gensim.models.Word2Vec.load('/tmp/mymodel')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
