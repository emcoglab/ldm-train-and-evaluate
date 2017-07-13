import glob
import os
import sys
import logging


logger = logging.getLogger()


def main():
    # Ignore files already processed and overwrite them?
    start_over = True

    subs_source_dir = "/Users/caiwingfield/Langboot local/Corpora/BBC/2 No nonspeech"
    subs_target_dir = "/Users/caiwingfield/Langboot local/Corpora/BBC/3 Replaced symbols"

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
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
