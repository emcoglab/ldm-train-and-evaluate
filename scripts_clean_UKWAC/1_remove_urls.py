import logging
import sys

from ..core.classes import SourceTargetPair, CorpusMetaData

logger = logging.getLogger()


def is_suspected_url_line(line):
    """
    Check if the line might be a url reference
    :param line:
    :return:
    """
    return line.startswith("CURRENT URL http://")


def main():
    corpus_meta = SourceTargetPair(
        source=CorpusMetaData(
            name="UKWAC",
            path="/Users/cai/Dox/Academic/Analyses/Corpus analysis/UKWAC/0 Raw untagged/cleaned_pre.pos.corpus"),
        target=CorpusMetaData(
            name="UKWAC",
            path="/Users/cai/Dox/Academic/Analyses/Corpus analysis/UKWAC/1 Text only/cleaned_pre.pos.corpus"))

    logger.info(f"Removing URL references from {corpus_meta.source.name} corpus")
    with open(corpus_meta.source.path, mode="r", encoding="utf-8", errors="ignore") as source_file:
        with open(corpus_meta.target.path, mode="w", encoding="utf-8") as target_file:
            i = 0
            for line in source_file:
                if is_suspected_url_line(line):
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
