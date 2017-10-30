"""
===========================
Looking various non-word tokens in a corpus.
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

import argparse
import logging
import re
import sys

import nltk

from ..core.corpus.tokenising import modified_word_tokenize
from ..core.corpus.filtering import filter_punctuation
from ..core.corpus.indexing import FreqDist

logger = logging.getLogger()


def main(corpus_dir, output_filename):
    """
    Outputs
    :param output_filename:
    :param corpus_dir:
    :return:
    """

    corpus_text = nltk.corpus.PlaintextCorpusReader(
        corpus_dir, ".+\..+")

    tokens = filter_punctuation(modified_word_tokenize(corpus_text.raw()))

    tokens = [token
              # only tokens which...
              for token in tokens
              # ...aren't alphanumeric...
              if not re.fullmatch("[A-Za-z0-9]+", token)]

    fd = FreqDist(tokens)

    mf = fd.most_common()

    with open(output_filename, mode="w", encoding="utf-8") as output_file:
        output_file.write(str(mf))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_dir")
    parser.add_argument("output_file")
    args = vars(parser.parse_args())

    main(args["corpus_dir"], args["output_file"])

    logger.info("Done!")
