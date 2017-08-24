"""
===========================
Converts all matrix market files to ndarray npz files in the specified directory.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2017
---------------------------
"""
import glob
import logging
import os
import sys

import numpy
import scipy.io

logger = logging.getLogger(__name__)


def main(directory_path: str):
    for mtx_filename in glob.glob(os.path.join(directory_path, "*.mtx")):
        logger.info(f"Loading .mtx file: '{mtx_filename}'")
        matrix = scipy.io.mmread(mtx_filename)
        npz_filename = os.path.splitext(mtx_filename)[0] + ".npz"
        logger.info(f"Saving as .npz file: '{npz_filename}'")
        numpy.savez(npz_filename, matrix)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main(sys.argv[1])
    logger.info("Done!")
