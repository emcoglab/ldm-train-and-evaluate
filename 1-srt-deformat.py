import glob
import os

import srt

from cw_common import *


def main():
    # Ignore files already processed and overwrite them?
    start_over = False

    raw_subs_dir       = "/Users/caiwingfield/Langboot local/Corpora/BBC/0 Raw"
    processed_subs_dir = "/Users/caiwingfield/Langboot local/Corpora/BBC/1 No srt formatting"

    subtitle_paths = list(glob.iglob(os.path.join(raw_subs_dir, '*.srt')))

    count = 0
    for subtitle_path in subtitle_paths:
        count += 1
        target_path = os.path.join(processed_subs_dir, os.path.basename(subtitle_path))

        # If we've already processed this file, skip it.
        if os.path.isfile(target_path) and not start_over:
            prints("{file_i:05d}: SKIPPING {file_name}".format(file_i=count, file_name=target_path))
            continue

        with open(subtitle_path, mode="r", encoding="utf-8", errors="ignore") as subtitle_file:
            sub_file_content = subtitle_file.read()

        subs = list(srt.parse(sub_file_content))

        with open(target_path, mode="w") as target_file:
            for sub in subs:
                target_file.write(sub.content + "\n")

        prints("{file_i:05d}: Processed {file_name}".format(file_i=count, file_name=target_path))


if __name__ == "__main__":
    main()
