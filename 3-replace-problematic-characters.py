import glob
import os
import string

from cw_common import *


def main():
    # Ignore files already processed and overwrite them?
    start_over = True

    subs_source_dir = "/Users/caiwingfield/Langboot local/Corpora/BBC/2 No nonspeech"
    subs_target_dir = "/Users/caiwingfield/Langboot local/Corpora/BBC/3 Replaced symbols"

    subs_source_paths = list(glob.iglob(os.path.join(subs_source_dir, '*.srt')))

    count = 0
    for source_path in subs_source_paths:
        count += 1
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


if __name__ == "__main__":
    main()
