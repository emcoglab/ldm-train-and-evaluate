import logging
import os
import sys

import numpy as np
import scipy.io as sio

from .preferences import Preferences
from ..core.utils.constants import Chiralities

logger = logging.getLogger()


def main():

    summed_dir = "/Users/caiwingfield/vectors/ngram"
    log_dir    = "/Users/caiwingfield/vectors/ngram_log"

    for meta in Preferences.source_corpus_metas:

        for radius in Preferences.window_radii:

            for chi in Chiralities:

                cooccur_filename = f"{meta.name}_r={radius}_{chi}.cooccur"

                # Load and add unsummed cooccurrence counts to get summed counts
                logger.info(f"Loading summed {chi}-cooccurrence matrix for radius {radius}")
                cooccur = sio.mmread(os.path.join(summed_dir, cooccur_filename)).tolil()

                # Apply log
                cooccur.data = np.log10(cooccur.data)

                # Save logged matrix
                logger.info(f"Saving {chi} log-ngram matrix for radius {radius}")
                sio.mmwrite(os.path.join(log_dir, cooccur_filename), cooccur)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
