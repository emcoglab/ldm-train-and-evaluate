"""
===========================
Sometimes subtitles of live broadcasts come in one word at a time.
This manifests in the subtitle file as a repeated entry with
one additional word each time, leading to lots of repeated sequences.
Here we can search for these to see if it's a big problem.
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

import srt

from ..ldm.preferences.preferences import Preferences


def main():
    raw_subs_dir = Preferences.bbc_processing_metas["raw"].path

    subtitle_paths = list(glob.iglob(os.path.join(raw_subs_dir, '*.srt')))

    for path in subtitle_paths:
        with open(path, mode="r", encoding="utf-8", errors="ignore") as subs_file:
            subs = list(srt.parse(subs_file.read()))

        last_line = ""
        for sub in subs:
            this_line = sub.content
            this_line.strip(" .")
            if (last_line != ""
                    and not last_line.isspace()
                    and last_line in this_line
                    and "#" not in this_line
                    and "-" not in this_line):
                print(f"{path}:\t***\t\"{last_line}\"\t⊑\t\"{this_line}\"")
            last_line = this_line


if __name__ == '__main__':
    main()
