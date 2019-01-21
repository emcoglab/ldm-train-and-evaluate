"""
===========================
The UKWAC is a *big* corpus, which can't be held in memory all at once.
Here we split it up into several "documents".
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

from ..core.utils.logging import log_message, date_format
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def main():

    # Need to make sure that we over-pad the numbering of the parts so that alphabetical order is numerical order
    target_filename_pattern = "part_{0:06d}.txt"
    lines_per_part = 10_000

    corpus_meta = dict(
        source=Preferences.ukwac_processing_metas["no_urls"],
        target=Preferences.ukwac_processing_metas["partitioned"])

    if not path.isdir(corpus_meta['target'].path):
        logger.warning(f"{corpus_meta['target'].path} does not exist, making it.")
        mkdir(corpus_meta['target'].path)

    logger.info(f"Loading {corpus_meta['source'].name} corpus from {corpus_meta['source'].path}")

    with open(corpus_meta['source'].path, mode="r", encoding="utf-8") as source_file:

        # initialise some variables
        part_number = 0
        lines = []
        total_line_count = 0

        while True:
            line = source_file.readline()
            if not line:
                break

            lines.append(line)

            if len(lines) >= lines_per_part:
                part_number += 1
                target_path = path.join(corpus_meta['target'].path, target_filename_pattern.format(part_number))

                total_line_count += len(lines)

                logger.info(f"Writing next {len(lines):,} lines to {path.basename(target_path)}."
                            f" ({total_line_count:,} lines total.)")

                with open(target_path, mode="w", encoding="utf-8") as target_file:
                    target_file.writelines(lines)

                # empty lines
                lines = []


if __name__ == '__main__':
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
