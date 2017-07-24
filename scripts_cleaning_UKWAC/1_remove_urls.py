import glob
import os
import sys
import logging

from ..core.classes import SourceTargetPair, CorpusMetaData

logger = logging.getLogger()


def is_suspected_url_line(line):
    return line.startswith("CURRENT URL http://")


def main():
    corpus_meta = SourceTargetPair(
        source=CorpusMetaData(
            name="UKWAC", path="/Users/caiwingfield/corpora/UKWAC/0 Raw untagged/"),
        target=CorpusMetaData(
            name="UKWAC", path="/Users/caiwingfield/corpora/UKWAC/1 Text only"))

    for source_path in glob.glob(os.path.join(corpus_meta.source.path, "*.*")):
        source_filename = os.path.basename(source_path)
        logger.info(f"Working on file {source_filename}")
        target_path = os.path.join(corpus_meta.target.path, source_filename)
        with open(source_path, mode="r", encoding="utf-8", errors="ignore") as source_file:
            with open(target_path, mode="w", encoding="utf-8") as target_file:
                i = 0
                for line in source_file:
                    if is_suspected_url_line(line):
                        # logger.info(f"Skipping line {line.strip()}")
                        continue
                    else:
                        target_file.write(line.strip() + "\n")
                        i += 1
                        if i % 100_000 == 0:
                            logger.info(f"Processed {i:,} lines")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
