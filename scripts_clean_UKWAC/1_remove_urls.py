"""
===========================
UKWAC documents are separated by the URLs from which they were retrieved.
We don't want these, so remove them here.
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

from ..core.utils.logging import log_message, date_format
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def is_suspected_url_ref(line):
    """
    Check if the line might be a url reference
    :param line:
    :return:
    """
    return line.startswith("CURRENT URL http://")


def main():
    corpus_meta = dict(
        source=Preferences.ukwac_processing_metas["raw"],
        target=Preferences.ukwac_processing_metas["no_urls"])

    logger.info(f"Removing URL references from {corpus_meta['source'].name} corpus")
    with open(corpus_meta['source'].path, mode="r", encoding="utf-8", errors="ignore") as source_file:
        with open(corpus_meta['target'].path, mode="w", encoding="utf-8") as target_file:
            i = 0
            for line in source_file:
                if is_suspected_url_ref(line):
                    continue
                else:
                    target_file.write(line.strip() + "\n")
                    i += 1
                    if i % 100_000 == 0:
                        logger.info(f"Processed {i:,} lines")


if __name__ == '__main__':
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
