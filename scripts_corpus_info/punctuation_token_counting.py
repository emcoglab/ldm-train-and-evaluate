import string
import re
import sys
import logging
import argparse

import nltk
import nltk.corpus as corpus

from ..core.tokenising import modified_word_tokenize
from ..core.filtering import ignorable_punctuation


logger = logging.getLogger()


def main(corpus_dir='/Users/caiwingfield/Langboot local/Corpora/Combined/0 SPEECH'):

    corpus_text = corpus.PlaintextCorpusReader(
        corpus_dir, ".+\..+")

    tokens = [token
              # only tokens which...
              for token in modified_word_tokenize(corpus_text.raw())
              # ...are only a single character long
              if len(token) == 1
              # ...are just made of punctuation
              or re.fullmatch('[' + string.punctuation + ']+', token)
              # ...are not a word-with-boring-punctuation
              or (not re.fullmatch('[A-Za-z0-9' + ignorable_punctuation + ']+', token))
              ]

    fd = nltk.probability.FreqDist(tokens)

    mf = fd.most_common()

    print(mf)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_dir")
    args = vars(parser.parse_args())

    main(args["corpus_dir"])

    logger.info("Done!")

