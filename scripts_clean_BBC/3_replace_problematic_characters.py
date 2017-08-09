"""
===========================
Replace badly encoded but potentially linguistically meaningful characters.
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
import os
import sys
import logging

from ..core.utils.logging import log_message, date_format
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def main():
    # Ignore files already processed and overwrite them?
    start_over = True

    subs_source_dir = Preferences.bbc_processing_metas["no_nonspeech"].path
    subs_target_dir = Preferences.bbc_processing_metas["replaced_symbols"]

    subs_source_paths = list(glob.iglob(os.path.join(subs_source_dir, '*.srt')))

    for i, source_path in enumerate(subs_source_paths):
        target_path = os.path.join(subs_target_dir, os.path.basename(source_path))

        # If we've already processed this file, skip it.
        if os.path.isfile(target_path) and not start_over:
            continue

        with open(source_path, mode="r", encoding="utf-8", errors="ignore") as source_file:
            with open(target_path, mode="w", encoding="utf-8") as target_file:
                for line in source_file:

                    fixed_line = str.replace(line, "�", "£")

                    # target_file.write(fixed_line.strip('\'\"-'))
                    target_file.write(fixed_line)

        if i % 1000 == 0:
            logger.info("Processed {count:02d} files".format(count=i))


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
