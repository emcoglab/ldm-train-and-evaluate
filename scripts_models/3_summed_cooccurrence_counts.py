import logging
import os
import sys

import scipy.io as sio
import scipy.sparse as sps

from .preferences import Preferences
from ..core.utils.indexing import TokenIndexDictionary

logger = logging.getLogger()


def main():

    unsummed_dir = "/Users/caiwingfield/vectors/n-gram"
    summed_dir = "/Users/caiwingfield/vectors/n-gram_summed"

    for meta in Preferences.source_corpus_metas:

        # We load the unsummed cooccurrence matrices in in sequence, and accumulate them to save the summed
        for radius in range(1, max(Preferences.window_radii) + 1):

            cooccur_filename_l = f"{meta.name}_r={radius}_left.cooccur"
            cooccur_filename_r = f"{meta.name}_r={radius}_right.cooccur"

            token_indices = TokenIndexDictionary.load(meta.index_path)
            vocab_size = len(token_indices)

            # Initialise cooccurrence matrices
            cooccur_l = sps.lil_matrix((vocab_size, vocab_size))
            cooccur_r = sps.lil_matrix((vocab_size, vocab_size))

            # Load and add unsummed cooccurrence counts to get summed counts
            logger.info(f"Loading unsummed cooccurrence matrix for radius {radius}")
            cooccur_l += sio.mmread(os.path.join(unsummed_dir, cooccur_filename_l))
            cooccur_r += sio.mmread(os.path.join(unsummed_dir, cooccur_filename_r))

            logger.info(f"Saving summed cooccurrence matrix for radius {radius}")
            sio.mmwrite(os.path.join(summed_dir, cooccur_filename_l), cooccur_l)
            sio.mmwrite(os.path.join(summed_dir, cooccur_filename_r), cooccur_r)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
