"""
===========================
Remove subtitle markup from .srt files, leaving only the subtitle content.
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
import sys
import logging
from os import path, mkdir

import srt

from ..ldm.utils.logging import log_message, date_format
from ..ldm.preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def main():
    # Ignore files already processed and overwrite them?
    start_over = False

    raw_subs_dir       = Preferences.bbc_processing_metas["raw"].path
    processed_subs_dir = Preferences.bbc_processing_metas["no_srt"].path

    if not path.isdir(processed_subs_dir):
        logger.warning(f"{processed_subs_dir} does not exist, making it.")
        mkdir(processed_subs_dir)

    subtitle_paths = list(glob.iglob(path.join(raw_subs_dir, '*.srt')))

    count = 0
    for subtitle_path in subtitle_paths:
        count += 1
        target_path = path.join(processed_subs_dir, path.basename(subtitle_path))

        # If we've already processed this file, skip it.
        if path.isfile(target_path) and not start_over:
            logger.info("{file_i:05d}: SKIPPING {file_name}".format(file_i=count, file_name=target_path))
            continue

        with open(subtitle_path, mode="r", encoding="utf-8", errors="ignore") as subtitle_file:
            sub_file_content = subtitle_file.read()

        subs = list(srt.parse(sub_file_content))

        with open(target_path, mode="w") as target_file:
            for sub in subs:
                target_file.write(sub.content + "\n")

        logger.info("{file_i:05d}: Processed {file_name}".format(file_i=count, file_name=target_path))


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
