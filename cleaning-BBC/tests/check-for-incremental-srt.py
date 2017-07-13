import glob
import os

import srt


def main():
    raw_subs_dir = "/Users/caiwingfield/Langboot local/Corpora/BBC/0 Raw"

    subtitle_paths = list(glob.iglob(os.path.join(raw_subs_dir, '*.srt')))

    for path in subtitle_paths:
        with open(path, mode="r", encoding="utf-8", errors="ignore") as subs_file:
            subs = list(srt.parse(subs_file.read()))

        last_line = ""
        for sub in subs:
            this_line = sub.content
            #this_line.replace("\n", " ")
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
