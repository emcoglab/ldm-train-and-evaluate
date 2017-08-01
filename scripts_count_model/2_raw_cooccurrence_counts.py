import logging
import os
import pickle
import sys

import matplotlib.pyplot as pplot
import scipy.sparse as sps

from ..core.classes import CorpusMetaData
from ..core.indexing import TokenIndexDictionary

logger = logging.getLogger()


def main():
    corpus_metas = [
        # dict(
        #     corpus=CorpusMetaData(
        #         name="toy",
        #         path="/Users/caiwingfield/corpora/toy-corpus/toy.corpus"),
        #     index_path="/Users/caiwingfield/vectors/indexes/toy.index",
        #     out_dir="/Users/caiwingfield/vectors/n-gram"
        # ),
        dict(
            corpus=CorpusMetaData(
                name="BBC",
                path="/Users/caiwingfield/corpora/BBC/4 Tokenised/BBC.corpus"),
            index_path="/Users/caiwingfield/vectors/indexes/BBC.index",
            out_dir="/Users/caiwingfield/vectors/n-gram"
        ),
        dict(
            corpus=CorpusMetaData(
                name="BNC",
                path="/Users/caiwingfield/corpora/BNC/2 Tokenised/BNC.corpus"),
            index_path="/Users/caiwingfield/vectors/indexes/BNC.index",
            out_dir="/Users/caiwingfield/vectors/n-gram"
        ),
        dict(
            corpus=CorpusMetaData(
                name="UKWAC",
                path="/Users/caiwingfield/corpora/UKWAC/3 Tokenised/UKWAC.corpus"),
            index_path="/Users/caiwingfield/vectors/indexes/UKWAC.index",
            out_dir="/Users/caiwingfield/vectors/n-gram"
        ),
    ]

    for corpus_meta in corpus_metas:

        for radius in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:

            logger.info(f"Working on {corpus_meta['corpus'].name} corpus, radius {radius}")

            # We will load in the full window and count left and right cooccurences separately
            # The size of the symmetric window is twice the radius, plus 1 (centre)
            diameter = 2 * radius + 1

            tid = TokenIndexDictionary.load(corpus_meta['index_path'])

            vocab_size = len(tid)

            # Indices
            lh_context_is = range(0, radius)
            target_i = radius
            rh_context_is = range(radius + 1, diameter)

            # first coordinate points to target word
            # second coordinate points to context word
            cooccur_l = sps.lil_matrix((vocab_size, vocab_size))
            cooccur_r = sps.lil_matrix((vocab_size, vocab_size))

            # Start scanning the corpus
            token_count = 0
            with open(corpus_meta['corpus'].path, mode="r", encoding="utf-8") as corpus_file:

                # Fill up the initial window, such that the next token to be read will produce the first full window
                window = []
                for i in range(0, diameter-1):
                    window.append(corpus_file.readline().strip())

                # Each iteration of this loop will advance the position of the window by one
                for corpus_token in corpus_file:

                    # Add a new token on the rhs of the window
                    window.append(corpus_token.strip())

                    # The window is now full
                    target_token = window[target_i]
                    target_id = tid.token2id[target_token]

                    # Count lh occurrences
                    for i in lh_context_is:
                        context_token = window[i]
                        context_id = tid.token2id[context_token]
                        cooccur_l[target_id, context_id] += 1

                    # Count rh occurrences
                    for i in rh_context_is:
                        context_token = window[i]
                        context_id = tid.token2id[context_token]
                        cooccur_r[target_id, context_id] += 1

                    # Pop the lhs token out of the window to await the next one, which will cause the window to have
                    # moved exactly one token over in the corpus
                    window = window[1:]

                    token_count += 1
                    if token_count % 1_000_000 == 0:
                        logger.info(f"\t{token_count:,} tokens processed")

            # logger.info(f"{cooccur_l.count_nonzero()} (L) "
            #             f"+ {cooccur_r.count_nonzero()} (R) "
            #             f"= {cooccur_l.count_nonzero() + cooccur_r.count_nonzero()} (total) "
            #             f"non-zeros in co-occurrence matrix")
            # logger.info(f"Summing to {int(cooccur_l.sum())} "
            #             f"+ {int(cooccur_r.sum())} "
            #             f"= {int(cooccur_l.sum() + cooccur_r.sum())} ")

            logger.info("Saving co-occurrence matrices")

            cooccur_lr = sps.lil_matrix((vocab_size, vocab_size))
            cooccur_lr += cooccur_l
            cooccur_lr += cooccur_r

            with open(os.path.join(corpus_meta['out_dir'], f"{corpus_meta['corpus'].name}_r={radius}_left"
                                                           f".cooccur"), mode="wb") as cooccur_file:
                pickle.dump(cooccur_l, cooccur_file)
            with open(os.path.join(corpus_meta['out_dir'], f"{corpus_meta['corpus'].name}_r={radius}_right"
                                                           f".cooccur"), mode="wb") as cooccur_file:
                pickle.dump(cooccur_r, cooccur_file)
            with open(os.path.join(corpus_meta['out_dir'], f"{corpus_meta['corpus'].name}_r={radius}_both"
                                                           f".cooccur"), mode="wb") as cooccur_file:
                pickle.dump(cooccur_lr, cooccur_file)

            # cooccur_lr = cooccur_lr.todense()
            #
            # logger.info("Saving heatmap")
            #
            # f = pplot.figure(num=None, figsize=(30, 30), dpi=192, facecolor='w', edgecolor='k')
            #
            # pplot.imshow(cooccur_lr, cmap="hot", interpolation='nearest')
            #
            # f.savefig(os.path.join(corpus_meta['out_dir'], f"{corpus_meta['corpus'].name}_r={radius}_heatmap.png"))
            #
            # pplot.close(f)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
