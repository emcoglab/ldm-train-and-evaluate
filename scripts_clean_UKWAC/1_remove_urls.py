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
from os import path, mkdir

from ..ldm.utils.logging import log_message, date_format
from ..ldm.preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def is_suspected_url_ref(line):
    """Check if the line might be a url reference"""
    return line.startswith("CURRENT URL http://")


def main():
    corpus_meta = dict(
        source=Preferences.ukwac_processing_metas["raw"],
        target=Preferences.ukwac_processing_metas["no_urls"])

    if not path.isdir(path.dirname(corpus_meta['target'].path)):
        logger.warning(f"{corpus_meta['target'].path} does not exist, making it.")
        mkdir(path.dirname(corpus_meta['target'].path))

    logger.info(f"Removing URL references from {corpus_meta['source'].name} corpus")
    with open(corpus_meta['source'].path, mode="r", encoding="utf-8", errors="ignore") as source_file:
        with open(corpus_meta['target'].path, mode="w", encoding="utf-8") as target_file:
            for line_i, line in enumerate(source_file):
                if is_suspected_url_ref(line):
                    continue
                else:
                    target_file.write(line.strip() + "\n")
                    if line_i % 100_000 == 0:
                        logger.info(f"Processed {line_i:,} lines")


if __name__ == '__main__':
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
