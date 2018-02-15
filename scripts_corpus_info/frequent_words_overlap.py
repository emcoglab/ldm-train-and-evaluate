"""
===========================
Assessing the overlap of most frequent words in eawch corpus.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2018
---------------------------
"""

import logging
import sys

from ..core.utils.logging import log_message, date_format
from ..core.corpus.indexing import FreqDist
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def main():

    top_sample = 300_000

    for corpus_1 in Preferences.source_corpus_metas:
        fd_1 = FreqDist.load(corpus_1.freq_dist_path)
        most_frequent_words_1 = set([word for word, _ in fd_1.most_common(top_sample)])
        for corpus_2 in Preferences.source_corpus_metas:
            if corpus_1.name == corpus_2.name:
                continue
            fd_2 = FreqDist.load(corpus_2.freq_dist_path)
            most_frequent_words_2 = set([word for word, _ in fd_2.most_common(top_sample)])

            percent_overlap = 100 * len(most_frequent_words_1.intersection(most_frequent_words_2)) / top_sample

            logger.info(f"Overlap between corpora {corpus_1.name} and {corpus_2.name}: {percent_overlap}%")


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    main()

    logger.info("Done!")
