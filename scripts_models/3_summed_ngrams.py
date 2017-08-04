import logging
import os
import sys

import scipy.io as sio
import scipy.sparse as sps

from .preferences import Preferences
from ..core.utils.indexing import TokenIndexDictionary
from ..core.utils.constants import Chiralities

logger = logging.getLogger()


def main():

    unsummed_dir = "/Users/caiwingfield/vectors/ngram_unsummed"
    summed_dir = "/Users/caiwingfield/vectors/ngram"

    for meta in Preferences.source_corpus_metas:

        token_indices = TokenIndexDictionary.load(meta.index_path)
        vocab_size = len(token_indices)

        # We load the unsummed cooccurrence matrices in in sequence, and accumulate them to save the summed
        for radius in range(1, max(Preferences.window_radii) + 1):

            for chi in Chiralities:

                cooccur_filename = f"{meta.name}_r={radius}_{chi}.cooccur"

                # Initialise cooccurrence matrices
                cooccur = sps.lil_matrix((vocab_size, vocab_size)).tolil()

                # Load and add unsummed cooccurrence counts to get summed counts
                logger.info(f"Loading unsummed {chi}-cooccurrence matrix for radius {radius}")
                cooccur += sio.mmread(os.path.join(unsummed_dir, cooccur_filename)).tolil()

                logger.info(f"Saving summed {chi}-cooccurrence matrix for radius {radius}")
                sio.mmwrite(os.path.join(summed_dir, cooccur_filename), cooccur)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
