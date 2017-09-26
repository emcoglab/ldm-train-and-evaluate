"""
===========================
Evaluate using priming data.
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

import logging
import sys

from ..core.evaluation.priming import SppData
from ..core.utils.logging import log_message, date_format


logger = logging.getLogger(__name__)


def main():

    spp_data = SppData()

    return


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
