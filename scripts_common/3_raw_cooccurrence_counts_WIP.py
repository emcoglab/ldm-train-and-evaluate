import logging
import os
import pickle
import sys

import nltk
import nltk.corpus
import numpy as np
import matplotlib.pyplot as plot


logger = logging.getLogger()


def main():

    corpus_dir = "/Users/caiwingfield/Langboot local/Corpora/toy-corpus/1 Tokenised and filtered"
    matrix_dir = "/Users/caiwingfield/Langboot local/Corpora/toy-corpus/2 Matrix"

    logger.info("Loading corpus")

    with open(os.path.join(corpus_dir, "corpus.p"), mode="rb") as corpus_file:
        corpus = pickle.load(corpus_file)

    with open(os.path.join(corpus_dir, "word2id.p"), mode="rb") as word2id_file:
        word2id = pickle.load(word2id_file)

    cooccur = np.zeros((len(word2id), len(word2id)))

    radius = 1

    # for a symmetric window of radius 1, we look at 1+2*1=3-grams (-1, 0, +1)
    n = 2 * radius + 1

    logger.info(f"For a symmetric radius-{radius} window, we're computing {n}-grams")

    grams = nltk.ngrams(corpus, n)

    for gram in list(grams):

        target_id = word2id.get(gram[radius], -1)

        if target_id == -1:
            logger.warning(f"Target word \"{gram[radius]}\" not found in index dictionary")
            continue

        for context_word_i in range(0, n):
            if context_word_i == radius:
                continue
            context_id = word2id.get(gram[context_word_i], -1)

            if context_word_i == -1:
                logger.warning(f"Context word \"{gram[context_word_i]}\" not found in index dictionary")
                continue

            cooccur[target_id, context_id] += 1

    logger.info("{total} total count, {nz} non-zeros in raw co-occurrence matrix"
                .format(total=cooccur.sum(), nz=np.count_nonzero(cooccur)))

    logger.info("Saving cooccurrence matrix")

    with open(os.path.join(matrix_dir, "cooccur.npy"), mode="wb") as matrix_file:
        np.save(matrix_file, cooccur)

    logger.info("Saving heatmap")

    f = plot.figure(num=None, figsize=(30,30), dpi=192, facecolor='w', edgecolor='k')

    plot.imshow(cooccur, cmap="hot", interpolation='nearest')

    f.savefig(os.path.join(matrix_dir, "heatmap.png"))

    plot.close(f)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
