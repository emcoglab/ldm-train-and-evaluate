import logging
import os
import pickle
import sys
from collections import defaultdict

import nltk
import numpy as np
import matplotlib.pyplot as pplot

from ..core.indexing import index_dictionary
from ..core.classes import CorpusMetaData

logger = logging.getLogger(__name__)


def main():

    index_dir = "/Users/caiwingfield/vectors/indexes"

    freq_dist_path = "/Users/caiwingfield/corpora/toy-corpus/info/Frequency distribution toy.corpus.pickle"

    corpus_name = "toy"

    with open(freq_dist_path, mode="rb") as freq_dist_file:
        freq_dist = pickle.load(freq_dist_file)

    word2id, id2word = index_dictionary(freq_dist)

    word2id_filename = os.path.join(index_dir, "{}_word2id.pickle".format(corpus_name))
    id2word_filename = os.path.join(index_dir, "{}_id2word.pickle".format(corpus_name))

    with open(word2id_filename, mode="wb") as word2id_file:
        pickle.dump(word2id, word2id_file)
    with open(id2word_filename, mode="wb") as id2word_file:
        pickle.dump(id2word, id2word_file)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
