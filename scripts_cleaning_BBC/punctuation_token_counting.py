import string
import re
import os
import sys
import logging

import nltk
import nltk.corpus as corpus

from ..core.modified_tokenizer import modified_word_tokenize
from ..core.ignorable_punctuation import ignorable_punctuation


logger = logging.getLogger()


def main():
    corpus_root = "/Users/caiwingfield/Langboot local/Corpora/BBC/3 Replaced symbols"

    corpus_text = corpus.PlaintextCorpusReader(
        corpus_root, ".*\.srt")

    tokens = [token
              # only tokens which...
              for token in modified_word_tokenize(corpus_text.raw())
              # ...are only a single character long
              if len(token) == 1
              # # ...are just made of punctuation
              # or re.fullmatch('[' + string.punctuation + ']+', token)
              # # ...are not a word-with-boring-punctuation
              # or (not re.fullmatch('[A-Za-z0-9' + ignorable_punctuation + ']+', token))
              ]

    fd = nltk.probability.FreqDist(tokens)

    mf = fd.most_common()

    print(mf)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")

