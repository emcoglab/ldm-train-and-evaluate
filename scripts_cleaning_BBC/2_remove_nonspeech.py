import glob
import os
import sys
import string
import logging


logger = logging.getLogger()


def is_suspected_nonspeech(line):
    """
    Checks if a line of subtitles looks like it might be a description of a visual event or other non-speech text
    :param line:
    :return:
    """

    # Strip leading and trailing punctuation and whitespace
    line_stripped = line.strip(string.punctuation + " \n")

    # A suspected stage direction is...
    return (
        # All uppercase
        line.isupper()
        # Not a permitted all-uppercase phrase (e.g. "I...")
        and line_stripped not in ['A', 'I', 'OK']
        # Doesn't contain a character which betrays this is probably speech ("GO TO HELL!")
        and not any(char in set("!?,.") for char in line))


def is_suspected_credits(line):
    """
    Checks of a line of subtitles looks like it might be subtitle credits
    :param line:
    :return:
    """

    # Strip leading and trailing punctuation and whitespace
    line_stripped = line.lower().strip(string.punctuation + " \n")

    return line_stripped.startswith("subtitles by ")
    pass


def main():
    # Ignore files already processed and overwrite them?
    start_over = True

    subs_source_dir = "/Users/caiwingfield/Langboot local/Corpora/BBC-mini/1 No srt formatting"
    subs_target_dir = "/Users/caiwingfield/Langboot local/Corpora/BBC-mini/2 No nonspeech"

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

                    # Skip nonspeech
                    if is_suspected_nonspeech(line) or is_suspected_credits(line):
                        # Log what we're skipping
                        print("{line}".format(line=line.strip(".()- \n")))
                        continue
                    else:
                        # strip of leading/trailing punctuation and whitespace, but add a newline
                        target_file.write(line.strip('\'\"-'))


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
