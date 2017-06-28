import glob
import os
import string

from cw_common import *


def is_suspected_stage_direction(line):
    # We look for expected stage directions

    # Strip leading and trailing punctuation and whitespace
    test_line = line.strip(string.punctuation + " \n")

    # A suspected stage direction is...
    return (
        # All uppercase
        test_line.isupper()
        # Not a permitted all-uppercase phrase (e.g. "I...")
        and test_line not in ['A', 'I', 'OK']
        # Doesn't contain a character which betrays this is probably speech ("GO TO HELL!")
        and not any(char in set("!?,.") for char in test_line))


def main():
    # Ignore files already processed and overwrite them?
    start_over = True

    subs_source_dir = "/Users/caiwingfield/Langboot local/Corpora/BBC/1 No srt formatting"
    subs_target_dir = "/Users/caiwingfield/Langboot local/Corpora/BBC/2 No metaspeech"

    subs_source_paths = list(glob.iglob(os.path.join(subs_source_dir, '*.srt')))

    count = 0
    for source_path in subs_source_paths:
        count += 1
        target_path = os.path.join(subs_target_dir, os.path.basename(source_path))

        # If we've already processed this file, skip it.
        if os.path.isfile(target_path) and not start_over:
            prints("{file_i:05d}: SKIPPING {file_name}".format(file_i=count, file_name=target_path))
            continue

        with open(source_path, mode="r", encoding="utf-8", errors="ignore") as source_file:
            with open(target_path, mode="w", encoding="utf-8") as target_file:
                for line in source_file:

                    # Skip stage directions
                    if is_suspected_stage_direction(line):
                        prints("    Skipping suspected stage direction: {line}".format(line=line.strip(" \n")))
                        continue
                    else:
                        # strip of leading/trailing punctuation and whitespace
                        target_file.write(line.strip('\'\" \n-'))

        prints("{file_i:05d}: Processed {file_name}".format(file_i=count, file_name=target_path))


if __name__ == "__main__":
    main()
