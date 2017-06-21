import glob
import os

import srt

from cw_common import *


def main():
    raw_subtitle_dir = "/Users/cai/Desktop/BBC Subtitle Corpus (raw 45k)"
    processed_subtitle_dir = "/Users/cai/Desktop/BBC Subtitle Corpus (no formatting)"

    subtitle_paths = list(glob.iglob(os.path.join(raw_subtitle_dir, '*.srt')))

    count = 0
    for subtitle_path in subtitle_paths:
        target_path = os.path.join(processed_subtitle_dir, os.path.basename(subtitle_path))
        with open(subtitle_path, mode="r", encoding="utf-8", errors="ignore") as subtitle_file:
            sub_file_content = subtitle_file.read()
        subs = list(srt.parse(sub_file_content))
        with open(target_path, mode="w") as target_file:
            for sub in subs:
                target_file.write(sub.content + "\n")
        count += 1
        prints("{file_i:05d}: Processed {file_name}".format(file_i=count, file_name=target_path))




if __name__ == "__main__":
    main()
