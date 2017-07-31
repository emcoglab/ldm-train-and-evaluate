import logging
import os
import pickle
import sys
from collections import defaultdict

import nltk
import numpy as np
import matplotlib.pyplot as pplot

from ..core.classes import CorpusMetaData

logger = logging.getLogger()


def main():

    corpus_meta = CorpusMetaData(
        name="toy",
        path="/Users/caiwingfield/corpora/toy-corpus/toy.corpus")

    freq_dist_path = "/Users/caiwingfield/corpora/toy-corpus/info/Frequency distribution toy.corpus.pickle"
    with open(freq_dist_path, mode="rb") as freq_dist_file:
        freq_dist = pickle.load(freq_dist_file)

    corpus_size = sum([freq for _, freq in freq_dist.most_common()])
    vocab_size  = len([freq for _, freq in freq_dist.most_common()])

    radius = 1
    # We will load in the full window and count left and right cooccurences separately
    # The size of the symmetric window is twice the radius, plus 1 (centre)
    diameter = 2 * radius + 1

    # Indices
    lh_context_is = range(0, radius)
    target_i = radius + 1
    rh_context_is = range(radius + 1, diameter + 1)

    coocur_l = defaultdict(
        default_factory=lambda: 0)
    with open(corpus_meta.path, mode="r", encoding="utf-8") as corpus_file:

        # Fill up the initial window, such that the next token to be read will produce the first full window
        window = corpus_file.readlines(diameter-1)

        # Read each iteration of this loop will advance the position of the window by one
        for token in corpus_file:

            # Add a new rh token
            window.append(token)

            # Count lh occurrences
            for i in lh_context_is:


            # Count rh occurrences


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

    f = pplot.figure(num=None, figsize=(30, 30), dpi=192, facecolor='w', edgecolor='k')

    pplot.imshow(cooccur, cmap="hot", interpolation='nearest')

    f.savefig(os.path.join(matrix_dir, "heatmap.png"))

    pplot.close(f)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
